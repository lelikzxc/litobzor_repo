"""Render the supervisor handout from its editable Markdown source."""
from pathlib import Path
import html
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE / 'issledovanie_metallizatsii.md'
OUTPUT = HERE / 'output/pdf/issledovanie_metallizatsii.pdf'
FONT = next((REPO / '.venv/lib').glob('python*/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('Text', str(FONT)))
INK = colors.HexColor('#202a32')
BLUE = colors.HexColor('#244d69')
WIDTH = A4[0] - 36 * mm
styles = {
    'body': ParagraphStyle('Body', fontName='Text', fontSize=9.1, leading=12.4, textColor=INK, spaceAfter=6),
    'title': ParagraphStyle('Title', fontName='Text', fontSize=17.5, leading=21.5, textColor=BLUE, spaceAfter=12),
    'heading': ParagraphStyle('Heading', fontName='Text', fontSize=11.6, leading=15, textColor=BLUE, spaceBefore=8, spaceAfter=5, keepWithNext=True),
    'subheading': ParagraphStyle('Subheading', fontName='Text', fontSize=10.2, leading=13.4, textColor=BLUE, spaceBefore=6, spaceAfter=4, keepWithNext=True),
    'cell': ParagraphStyle('Cell', fontName='Text', fontSize=8.3, leading=11, textColor=INK),
    'caption': ParagraphStyle('Caption', fontName='Text', fontSize=8.1, leading=10.6, textColor=colors.HexColor('#4b5862'), spaceAfter=6),
}


def paragraph(text, style='body'):
    value = html.escape(text)
    value = re.sub(r'(https?://[^\s]+)', r'<link href="\1" color="#244d69">\1</link>', value)
    return Paragraph(value, styles[style])


def table(lines):
    raw = [[c.strip() for c in line.strip().strip('|').split('|')] for line in lines]
    raw = [row for row in raw if not all(re.fullmatch(r':?-+:?', cell or '') for cell in row)]
    n = len(raw[0]); widths = [WIDTH*.43, WIDTH*.57] if n == 2 else (
        [WIDTH*.46, WIDTH*.27, WIDTH*.27] if n == 3 else [WIDTH*.40, WIDTH*.20, WIDTH*.20, WIDTH*.20])
    if n == 3 and raw[0][0] == 'Этап': widths = [WIDTH*.19, WIDTH*.57, WIDTH*.24]
    cells = [[paragraph(cell, 'cell') for cell in row] for row in raw]
    value = Table(cells, colWidths=widths, hAlign='LEFT', repeatRows=1)
    value.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f4')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
        ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#b8c8d3')),
        ('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#dce2e6')),
    ]))
    return value


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#c6d0d7'))
    canvas.line(18*mm, 13*mm, A4[0]-18*mm, 13*mm)
    canvas.setFont('Text',8); canvas.setFillColor(colors.HexColor('#64717b'))
    canvas.drawRightString(A4[0]-18*mm, 8.5*mm, str(doc.page))
    canvas.restoreState()


def main():
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    story=[]; lines=SOURCE.read_text().splitlines(); i=0
    while i < len(lines):
        text=lines[i].strip()
        if not text: i+=1; continue
        if text == '<!-- pagebreak -->': story.append(PageBreak());i+=1;continue
        if text.startswith('# '):story.append(paragraph(text[2:],'title'));i+=1;continue
        if text.startswith('## '):story.append(paragraph(text[3:],'heading'));i+=1;continue
        if text.startswith('### '):story.append(paragraph(text[4:],'subheading'));i+=1;continue
        if text.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i]);i+=1
            story.extend([table(block),Spacer(1,7)]);continue
        image=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',text)
        if image:
            path=(HERE/image.group(2)).resolve()
            value=Image(str(path));ratio=value.imageHeight/value.imageWidth
            value.drawWidth=WIDTH;value.drawHeight=WIDTH*ratio
            story.extend([Spacer(1,3),value,paragraph(image.group(1),'caption')]);i+=1;continue
        block=[text];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','|','![','<!--')):
            block.append(lines[i].strip());i+=1
        # Numbered experimental steps stay individual paragraphs.
        if all(re.match(r'\d+\. ',line) for line in block):
            story.extend(paragraph(line) for line in block)
        else:story.append(paragraph(' '.join(block)))
    doc=SimpleDocTemplate(str(OUTPUT),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,
        topMargin=15*mm,bottomMargin=19*mm,title='Сегментация SEM-изображений и классификация геометрии металлизации',
        author='',subject='Данные, архитектура CNN, протокол сравнений и результаты',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUTPUT)


if __name__=='__main__':main()
