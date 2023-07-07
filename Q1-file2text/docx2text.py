import docx

def extract_text(element):
    if isinstance(element, docx.text.paragraph.Paragraph):
        return element.text
    elif isinstance(element, docx.table.Table):
        text = ''
        for row in element.rows:
            for cell in row.cells:
                text += extract_text(cell)
        return text
    elif isinstance(element, docx.oxml.ns.OLEObject):
        return element.text
    else:
        return ''

def read_docx(file_path):
    doc = docx.Document(file_path)
    text = ''
    for element in doc.element.body:
        text += extract_text(element)
    return text

file_path = './CV/1.docx'
result = read_docx(file_path)
print(result)
