import os
import concurrent.futures
import json
import urllib.request
import urllib.error as er
import ssl
from tqdm import tqdm

def pdf_download(pdf_url, save_dir: str, num):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{num}.pdf")

    try:
        # Open the URL and read the response
        with urllib.request.urlopen(pdf_url, timeout=10) as response:
            # Check if the response is a PDF
            content_type = response.getheader('Content-Type')
            if 'application/pdf' not in content_type:
                return f"Error: URL did not return a PDF for {num}"

            # Write the content to a file
            with open(output_path, 'wb') as file:
                file.write(response.read())
        
        return f"Successfully downloaded {num}"

    except er.URLError: 
        ssl_context = ssl._create_unverified_context()
        
        try:
            # Open the URL and read the response
            with urllib.request.urlopen(pdf_url, context=ssl_context, timeout=5) as response:
                # Check if the response is a PDF
                content_type = response.getheader('Content-Type')
                if 'application/pdf' not in content_type:
                    return f"Error: URL did not return a PDF for {num}"
                # Write the content to a file
                with open(output_path, 'wb') as file:
                    file.write(response.read())
            return f"Successfully downloaded {num}"

        except Exception as e:
            print(f"Disabling ssl verification did not work for {num}")

    except urllib.error.HTTPError as e:
        return f"HTTP error downloading PDF for {num}: {str(e)}"
    except Exception as e:
        return f"Error downloading PDF for {num}: {str(e)}"

def download_wrapper(args):
    return pdf_download(*args)

def main():
    with open(DOWNLOAD_FOLDER_PATH, "r", encoding="utf-8") as file:
        content = json.load(file)

    download_args = [(link, PDF_PATH, num) for num, link in content.items()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(download_wrapper, download_args), total=len(download_args), desc="Downloading PDFs"))

    # Print results
    for result in results:
        print(result)

    print(f"All downloads attempted, saved at '{PDF_PATH}'")

if __name__ == "__main__":
    DOWNLOAD_FOLDER_PATH = 'AI-Internship-Task-main/Dataset.json'  # Use forward slashes for compatibility
    PDF_PATH = 'PDFs'
    main()