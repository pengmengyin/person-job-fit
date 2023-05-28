import PyPDF2
import os
from docx2pdf import convert
from ecloud import CMSSEcloudOcrClient
import json

accesskey = 'c9f9e00293c247649c92e7b00be8fa47'
secretkey = '9ae7550615734b2c80b16434a24ced84'
url = 'https://api-wuxi-1.cmecloud.cn:8443'

def request_webimage_file(imagepath):
    print("请求File参数")
    requesturl = '/api/ocr/v1/webimage'

    try:
        ocr_client = CMSSEcloudOcrClient(accesskey, secretkey, url)
        response = ocr_client.request_ocr_service_file(requestpath=requesturl, imagepath=imagepath)

        data = json.loads(response.text)
        if 'body' in data:
            prism_wordsInfo = data['body']['content']['prism_wordsInfo']
            for i in prism_wordsInfo:
                print(i['word'])

    except ValueError as e:
        print(e)

def pdf2text(path):
    str = ''
    # Open the PDF file in read-binary mode
    with open(path, 'rb') as pdf_file:

        # Create a PdfReader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Get the number of pages in the PDF file
        num_pages = len(pdf_reader.pages)

        # Loop through each page and extract the text
        for page_num in range(num_pages):
            # Get the page object
            page_obj = pdf_reader.pages[page_num]

            # Extract the text from the page object
            page_text = page_obj.extract_text()
            #字符串拼接
            str = str + page_text
            # Do something with the text
            # ...

    pdf_file.close()
    return str.replace('\n', '')

def docx2text(folder_path):
    str = ''
    try:
        new_file_name = folder_path.replace("/CV", "").replace("docx", "pdf")
        # 将 DOCX 文件转换为 PDF 文件，并保存到与 DOCX 文件相同的路径下
        convert(folder_path, './')
        str = str + pdf2text(new_file_name)
        str = str + os.linesep+ os.linesep
        os.remove(new_file_name)
    except Exception as e:
        print(e)
    return str


# file_path =r'./images/3.png'
file_path='./CV/1.docx'
# file_path = './CV/1.pdf'
if file_path.endswith('.jpg') or file_path.endswith('.png') :
    results = request_webimage_file(file_path)
elif file_path.endswith('.pdf'):
    results = pdf2text(file_path)
elif file_path.endswith('.docx'):
    results = docx2text(file_path)
else:
    print('输入文件错误')
print(results)

