from pathlib import Path
import json,hashlib
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];p=R/'Jounral_PIWM/elsarticle-template-num.pdf';out=R/'reports/corrected_query/manuscript_render';out.mkdir(parents=True,exist_ok=True)
doc=pdfium.PdfDocument(str(p));reader=PdfReader(p);thumbs=[];selected=[]
for i in range(len(doc)):
    im=doc[i].render(scale=1.2).to_pil().convert('RGB')
    text=reader.pages[i].extract_text() or ''
    if i==0 or any(k in text for k in ['Table 1:','Table 2:','Table 3:','Table 4:','Table 5:','Table 6:','Table 7:','unit-speed road','Implemented prediction pathways','Complete conditioned-dynamics','Chronological-holdout errors','Corrected-label comparison','Chronological initialization sensitivity']):
        doc[i].render(scale=1.7).to_pil().save(out/f'final_page_{i+1:02d}.png');selected.append(i+1)
    im.thumbnail((300,424));tile=Image.new('RGB',(320,454),'#eeeeee');tile.paste(im,((320-im.width)//2,20));ImageDraw.Draw(tile).text((8,4),f'Page {i+1}',fill='black');thumbs.append(tile)
for begin in range(0,len(thumbs),12):
    sheet=Image.new('RGB',(1280,1362),'white')
    for j,t in enumerate(thumbs[begin:begin+12]):sheet.paste(t,((j%4)*320,(j//4)*454))
    sheet.save(out/f'final_contact_{begin//12+1}.png')
obj=dict(pages=len(doc),pdf_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),selected_pages=selected,rendered_all_pages=True,visual_review='pending')
(R/'reports/corrected_query/manuscript_render_manifest.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8');print(obj)
