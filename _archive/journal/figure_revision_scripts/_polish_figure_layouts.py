from pathlib import Path
p=Path('scripts/prepare_conference_assets.py');s=p.read_text(encoding='utf-8');s=s[:s.index('# Recompose six')]+r'''
# Recompose six known-parameter panels; preserve the original graphics and
# ground-truth overlays. Source numeric tick positions are retained exactly.
import numpy as np
import pdfplumber
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont('CMR',str(ROOT/'.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/cmr10.ttf')))
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
buf=io.BytesIO();c=canvas.Canvas(buf,pagesize=(W,H));placements=[]
for i,(title,num,ticktop) in enumerate(panels):
    raw=np.array(PdfReader(SRC/'allpara.pdf').pages[0].images[f'Image{num}'].image.convert('RGB'))
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
for box,x,y,scale in placements:
    part=copy.copy(source);part.mediabox.lower_left=(box[0],box[1]);part.mediabox.upper_right=(box[2],box[3]);part.cropbox=part.mediabox
    page.merge_transformed_page(part,Transformation().translate(-box[0],-box[1]).scale(scale).translate(x,y))
writer=PdfWriter();writer.add_page(page)
with open(ROOT/'Jounral_PIWM/imgs/fig_conference_parameters.pdf','wb') as stream:writer.write(stream)
print('Exact conference assets and parameter panel layouts prepared.')
''';p.write_text(s,encoding='utf-8')
p=Path('scripts/fig_main_cv.py');s=p.read_text(encoding='utf-8').replace('[.11, .47, .86, .35]','[.11, .50, .86, .32]');p.write_text(s,encoding='utf-8')
p=Path('scripts/fig_conference.py');s=p.read_text(encoding='utf-8').replace('else .495','else .480');p.write_text(s,encoding='utf-8')
p=Path('Jounral_PIWM/elsarticle-template-num.tex');s=p.read_text(encoding='utf-8')
s=s.replace(r'{Fixed curvature preview\\$\hat\kappa(\Delta),\quad 0\leq\Delta\leq4.5$ m}',r'{Curvature preview\\$\hat\kappa(\Delta)$\\$\Delta\in[0,4.5]$ m}')
s=s.replace(r'Ground-truth state at $t_0$\\for the reported evaluation',r'GT initialization at $t_0$')
s=s.replace(r'Structured transition\\Vehicle: learned actuation\\Road: Frenet geometry',r'Split dynamics\\Vehicle actuation\\Frenet geometry')
s=s.replace(r'at (4.35,1.65) {Action $a_t$}',r'at (4.35,1.2) {Action $a_t$}')
s=s.replace(r'(4.35,1.4) |-',r'(4.35,.98) |-')
s=s.replace(r'Update $d$ and $e_\psi$ with Frenet geometry.',r'Update $d$ and $e_\psi$\\with Frenet geometry.')
s=s.replace('with three evenly\nspaced display columns selected','with three\ndisplay columns selected')
s=s.replace(r'An \emph{autonomous CPS} $s = (X, I, Y, A, \phi_{\theta}, g, h)$ models the',r'An \emph{autonomous CPS}, represented by the tuple $s = (X, I, Y, A, \phi_{\theta}, g, h)$, models the')
s=s.replace(r'\begin{algorithm}[t]',r'\begin{algorithm}[p]')
p.write_text(s,encoding='utf-8')
