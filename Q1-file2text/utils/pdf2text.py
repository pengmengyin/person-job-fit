import PyPDF2


def pdf2text(path):
    str = ''
    with open(path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page_obj = pdf_reader.pages[page_num]
            page_text = page_obj.extract_text()
            #字符串拼接
            str = str + page_text


    pdf_file.close()
    return str.replace('\n', '')

def split_sentence(sentence, max_len=450):
    """
    Split a sentence into multiple parts, each with a maximum length of max_len.
    """
    words = sentence.split()
    parts, current_part = [], words[0]
    for word in words[1:]:
        if len(current_part) + 1 + len(word) > max_len:  # add 1 for the space
            parts.append(current_part)
            current_part = word
        else:
            current_part += ' ' + word
    parts.append(current_part)
    return parts
