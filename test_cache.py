import requests
import time
import os

BASE_URL = "http://localhost:5000/api"
TEST_FILE = r"c:\Users\Usuário\OneDrive\Documentos\Agentes de IA\ai-doc-verifier - v2\uploads\20260208_145354_bl_6._Draft_BL.pdf"

def upload_file(filepath):
    with open(filepath, 'rb') as f:
        files = {'file': f}
        data = {'doc_type': 'bl'}
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
        return response.json()

def extract_data(filepath, doc_type):
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/extract", json={
        'filepath': filepath,
        'doc_type': doc_type
    })
    duration = time.time() - start_time
    return response.json(), duration

def main():
    print(f"Testing with file: {TEST_FILE}")
    
    # 1. First Upload & Extract
    print("\n--- Run 1 (Should be uncached) ---")
    upload_res1 = upload_file(TEST_FILE)
    if 'error' in upload_res1:
        print(f"Upload failed: {upload_res1['error']}")
        return
        
    uploaded_path1 = upload_res1['path']
    print(f"Uploaded to: {uploaded_path1}")
    
    data1, duration1 = extract_data(uploaded_path1, 'bl')
    print(f"Extraction 1 took: {duration1:.2f} seconds")
    if data1.get('success'):
        print("Success!")
    else:
        print(f"Failed: {data1.get('error')}")

    # 2. Second Upload & Extract (Same content, different file on server)
    print("\n--- Run 2 (Should be cached) ---")
    upload_res2 = upload_file(TEST_FILE)
    uploaded_path2 = upload_res2['path']
    print(f"Uploaded to: {uploaded_path2}")
    
    data2, duration2 = extract_data(uploaded_path2, 'bl')
    print(f"Extraction 2 took: {duration2:.2f} seconds")
    if data2.get('success'):
        print("Success!")
    else:
        print(f"Failed: {data2.get('error')}")

    # Validation
    print("\n--- Results ---")
    if duration2 < duration1 * 0.5: # Expect significant speedup
        print("PASS: Cache likely worked (Run 2 was much faster)")
    else:
        print("FAIL: Run 2 was not significantly faster. Cache might not be working.")
        
    if data1.get('data') == data2.get('data'):
         print("PASS: Data matches between runs")
    else:
         print("FAIL: Data mismatch")

if __name__ == "__main__":
    main()
