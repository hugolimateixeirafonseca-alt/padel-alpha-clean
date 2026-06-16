import base64
import io

import requests
from flask import Flask, request, jsonify
from PIL import Image, ImageFilter

app = Flask(__name__)


@app.get("/")
def health():
    # Render usa isto para o health check; tambem te serve para testar no browser
    return "ok"


@app.post("/clean")
def clean():
    """
    Recebe JSON:
      {
        "url": "<URL do PNG transparente do BiRefNet>",
        "imgbb_key": "<chave ImgBB>",
        "threshold": 128,   # opcional: limiar do alfa (0-255)
        "erode": 1          # opcional: pixeis de borda a aparar (0-3)
      }
    Devolve:
      { "url": "<URL direto do PNG limpo no ImgBB>" }
    """
    try:
        data = request.get_json(force=True) or {}
        url = data["url"]
        imgbb_key = data["imgbb_key"]
        threshold = int(data.get("threshold", 128))
        erode = max(0, min(3, int(data.get("erode", 1))))

        # 1) descarregar o PNG do BiRefNet
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")

        # 2) binarizar o alfa: cada pixel fica 0% ou 100% transparente (sem meios-tons)
        r, g, b, a = img.split()
        a = a.point(lambda v: 255 if v >= threshold else 0)

        # 3) erodir N pixeis: encolhe a zona opaca e remove a franja clara das bordas
        #    (e tambem apaga pequenos pontos isolados)
        for _ in range(erode):
            a = a.filter(ImageFilter.MinFilter(3))

        img.putalpha(a)

        # 4) gravar PNG em memoria
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()

        # 5) upload ao ImgBB e devolver o URL direto
        up = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": imgbb_key, "image": b64, "name": "padel_clean"},
            timeout=120,
        )
        up.raise_for_status()
        out_url = up.json()["data"]["url"]
        return jsonify({"url": out_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
