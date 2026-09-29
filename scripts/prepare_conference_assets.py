"""Prepare exact conference assets. Run from piwm/ with Python + pypdf/reportlab.
The plot rasters are embedded originals; no curve is digitized or regenerated.
"""
from pathlib import Path
import hashlib, json, io, copy
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'_archive/conference_source/imgs'
OUT=ROOT/'_archive/conference_source/extracted'
OUT.mkdir(parents=True,exist_ok=True)
(ROOT/'Jounral_PIWM/imgs').mkdir(parents=True,exist_ok=True)
manifest={}
for name in ['resultsx2','cartresults','allpara','pred','cartviz']:
    reader=PdfReader(SRC/(name+'.pdf'))
    for asset in reader.pages[0].images:
        dest=OUT/(name+'_'+asset.name)
        dest.write_bytes(asset.data)
        manifest[dest.name]={'source':'imgs/'+name+'.pdf','object':asset.name,
            'size':asset.image.size,'sha256':hashlib.sha256(asset.data).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

# Recompose six known-parameter panels; preserve the original graphics and
# ground-truth overlays. Source numeric tick positions are retained exactly.
import numpy as np
import pdfplumber
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
font=ROOT/'.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/cmr10.ttf'
if not font.exists():
    import matplotlib
    font=Path(matplotlib.get_data_path())/'fonts/ttf/cmr10.ttf'
pdfmetrics.registerFont(TTFont('CMR',str(font)))
source=PdfReader(SRC/'allpara.pdf').pages[0]
matrices={}
def image_matrix(op,args,cm,tm):
    if op==b'Do':matrices[str(args[0]).lstrip('/')]=[float(x) for x in cm]
source.extract_text(visitor_operand_before=image_matrix)
with pdfplumber.open(SRC/'allpara.pdf') as pdf:
    words=pdf.pages[0].extract_words()
panels=[('Lunar Lander: main-engine power',19,132),('Lunar Lander: side-engine power',20,132),
        ('CartPole: pole mass',15,272),('CartPole: pole length',16,272),
        ('CartPole: cart length',18,272),('CartPole: applied force',17,272)]
W,H=388.54,415
buf=io.BytesIO();c=canvas.Canvas(buf,pagesize=(W,H),initialFontName='CMR');placements=[]
for i,(title,num,ticktop) in enumerate(panels):
    raw=np.array(PdfReader(SRC/'allpara.pdf').pages[0].images[f'/Image{num}'].image.convert('RGB'))
    h,w=raw.shape[:2];dark=(raw.max(2)<155)
    xl=int(dark[:,:30].sum(0).argmax());xr=w-30+int(dark[:,-30:].sum(0).argmax())
    yt=int(dark[:20,:].sum(1).argmax());yb=h-20+int(dark[-20:,:].sum(1).argmax())
    a,_,_,d,e,f=matrices[f'Image{num}']
    box=(e+a*xl/w, f+d*(1-yb/h), e+a*(xr+1)/w, f+d*(1-(yt-1)/h))
    row,col=divmod(i,2);left=col*198;top=H-row*137
    scale=146/(box[2]-box[0]);x=left+35;y=top-124
    c.setFillColorRGB(.12,.12,.12);c.setFont('CMR',9)
    c.drawString(left+2,top-9,f'({chr(97+i)}) {title}')
    ticks=[z for z in words if abs(z['top']-ticktop)<2 and all(ch in '0123456789.' for ch in z['text'])
           and e-10<(z['x0']+z['x1'])/2<e+a+8]
    c.setFont('CMR',9)
    for z in ticks:
        cx=(z['x0']+z['x1'])/2
        c.drawCentredString(x+(cx-box[0])*scale,y-12,z['text'])
    # Label the center of each original colored interval, not an inferred value.
    R,G,B=[raw[:,:,k].astype(float) for k in range(3)]
    masks=[(B>R+30)&(G>R+20), (R>G+20)&(abs(G-B)<25)&(B<170), (R>G+40)&(B>G+30)]
    for mask,lab in zip(masks,['0%','5%','10%']):
        py=float(np.median(np.where(mask)[0]));sy=f+d*(1-py/h)
        c.drawRightString(left+29,y+(sy-box[1])*scale-3,lab)
    placements.append((box,x,y,scale))
c.save();page=PdfReader(buf).pages[0]
from pypdf.generic import ContentStream,NameObject
stream=ContentStream(source.get_contents(),source.pdf)
kept=[];in_text=False
for operands,op in stream.operations:
    if op==b'BT':in_text=True
    if not in_text:kept.append((operands,op))
    if op==b'ET':in_text=False
stream.operations=kept
source[NameObject('/Contents')]=stream
source['/Resources'].pop('/Font',None)
for box,x,y,scale in placements:
    part=copy.copy(source);part.mediabox.lower_left=(box[0],box[1]);part.mediabox.upper_right=(box[2],box[3]);part.cropbox=part.mediabox
    page.merge_transformed_page(part,Transformation().translate(-box[0],-box[1]).scale(scale).translate(x,y))
writer=PdfWriter();writer.add_page(page)
with open(ROOT/'Jounral_PIWM/imgs/fig_conference_parameters.pdf','wb') as stream:writer.write(stream)
print('Exact conference assets and parameter panel layouts prepared.')
