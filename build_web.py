# weights.json + web/engine.js + web/template.html → omicron.html (태블릿/폰 브라우저용)
# 실행: python export_web.py && python build_web.py
html = open("web/template.html", encoding="utf-8").read()
weights = open("weights.json", encoding="utf-8").read().replace("</", "<\\/")
engine = open("web/engine.js", encoding="utf-8").read()
html = html.replace("__WEIGHTS__", weights).replace("__ENGINE__", engine)
open("omicron.html", "w", encoding="utf-8").write(html)
print("omicron.html 생성")
