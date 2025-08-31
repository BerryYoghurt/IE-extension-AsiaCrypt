import http.server
import socketserver
from io import BytesIO
import os
# from multipart import MultipartParser
from requests_toolbelt.multipart import decoder
from PIL import Image
from image_encryption.jpeg import decrypt, gen_keys

PORT = 8000

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        print(self.headers)
        content_type = self.headers['Content-Type']
        # boundary = content_type.split("boundary=")[1].encode()
        # print("Boundary is ", boundary)

        try:
            # Read the content length to get the size of the incoming data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            print(post_data[:100])
            
            # Parse the data
            multipart_data = decoder.MultipartDecoder(post_data, content_type)
            password = multipart_data.parts[0].text.strip()
            # print("Password is ", password)
            image = multipart_data.parts[1].content

            # Process the image
            with open('ctxt.jpeg', 'wb') as f:
                f.write(BytesIO(image).read())
            ret, image = decrypt(password, True)

            output_buffer = BytesIO()
            image.save(output_buffer, format='JPEG', quality=90)
            # image.save("rotated.jpeg")
            image.save("decrypted.jpeg", quality=90)
            output_buffer.seek(0)
            
            # Send response
            if ret:
            # Authentication succeeded
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(output_buffer.read())
            else:
                # Authentication failed
                self.send_response(406)
                self.end_headers()
                self.wfile.write(b'Authentication Failed')
                # pass
        except IOError:
            # Handle IO error and send a "malformed request" response
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Malformed request')
        except BaseException as e:
            print(e)
            exit(-1)

with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"Serving on port {PORT}")
    httpd.serve_forever()
