
import requests
import config
import certifi
from PIL import Image
from io import BytesIO

def main():
    print("\n========== MAPILLARY DOWNLOAD TEST ==========\n")

    params = {
        "access_token": config.MAPILLARY_TOKEN,
        "fields": "id,thumb_1024_url,computed_geometry,is_pano",
        "bbox": "76.4000,10.9500,76.4600,11.0200",
        "limit": 100
    }

    try:
        response = requests.get(
            "https://graph.mapillary.com/images",
            params=params,
            timeout=30,
            verify=certifi.where()
        )
        response.raise_for_status()

        images = response.json().get("data", [])

        if not images:
            print("No images found.")
            return

        sample = next(
            (x for x in images
             if x.get("is_pano") and x.get("thumb_1024_url")),
            None
        )

        if not sample:
            print("No panorama thumbnail available.")
            return

        print("Image ID:", sample["id"])
        print("Downloading thumbnail...")

        image_response = requests.get(
            sample["thumb_1024_url"],
            timeout=30,
            verify=certifi.where()
        )
        image_response.raise_for_status()

        image = Image.open(BytesIO(image_response.content))
        image.save("mapillary_sample.jpg")

        print("Image dimensions:", image.size)
        print("Saved successfully: mapillary_sample.jpg")

    except Exception as e:
        print("\nERROR:", type(e).__name__, str(e))


if __name__ == "__main__":
    main()
