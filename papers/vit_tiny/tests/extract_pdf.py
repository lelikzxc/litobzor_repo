"""Extract text from vit.pdf page by page."""
import fitz
doc = fitz.open('vit.pdf')
for i in range(doc.page_count):
    page = doc[i]
    text = page.get_text()
    with open(f'vit_page_{i+1}.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Page {i+1}: {len(text)} chars')
print('DONE')