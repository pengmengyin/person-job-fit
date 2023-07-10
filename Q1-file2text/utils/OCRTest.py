from ecloud import CMSSEcloudOcrClient
import json
accesskey = 'c9f9e00293c247649c92e7b00be8fa47'
secretkey = '9ae7550615734b2c80b16434a24ced84'
url = 'https://api-wuxi-1.cmecloud.cn:8443'
def request_webimage_file():
    print("请求File参数")
    requesturl = '/api/ocr/v1/webimage'
    imagepath = r'./images/3.png'
    try:
        ocr_client = CMSSEcloudOcrClient(accesskey, secretkey, url)
        response = ocr_client.request_ocr_service_file(requestpath=requesturl, imagepath= imagepath)

        data = json.loads(response.text)
        if 'body' in data:
            prism_wordsInfo = data['body']['content']['prism_wordsInfo']
            for i in prism_wordsInfo:
                print(i['word'])
        
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    request_webimage_file()

