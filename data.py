from pathlib import Path



def collect_images(folder_path):

    image_list = []


    for file_path in folder_path.glob("*.png"):
        if file_path.stem[-1] == '1' :
            tb = 1
        else:
            tb = 0
        image_data = (file_path, tb)
        image_list.append(image_data)

    return image_list




shenzhen_images = collect_images(Path("data/Shenzhen/images/images"))
montgomery_images = collect_images(Path("data/Montgomery/images/images"))

print(len(shenzhen_images))
print(len(montgomery_images))
