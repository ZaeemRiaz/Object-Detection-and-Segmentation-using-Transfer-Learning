import glob
import xml.etree.ElementTree as ET

from PIL import Image
from bbaug import policies
from numpy import asarray

class_name_dict = {
    0: 'First Name',
    1: 'Last Name',
    2: 'Country',
    3: 'ID Number',
    4: 'DOB'
}


def generate_new_name(temp):
    return str(int(temp) + 1)


start = '28'
path = 'Dataset/pak cnic/images'
for img_file in glob.glob(path + '/*.jpg'):
    image = Image.open(img_file)
    img = asarray(image)
    xml_path = img_file[:-4]
    label_file_read = xml_path + '.xml'
    tree = ET.parse(label_file_read)
    root = tree.getroot()

    class_name_list = []
    bnd_box_list = []

    # read filename
    filename = root.find('filename').text

    # read size
    sizeTag = root.find('size')
    size = sizeTag.find('width').text, sizeTag.find('height').text, sizeTag.find('depth').text

    for objectTag in root.findall('object'):
        # read class name
        class_name = objectTag.find('name').text

        # read bndboxes
        box = objectTag.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)

        # store in lists

        class_name_list.append(class_name)
        bnd_box_list.append([xmin, ymin, xmax, ymax])

    class_name_number = []
    for class_name in class_name_list:
        for key, value in class_name_dict.items():
            if class_name == value:
                class_name_number.append(key)

    for num in range(0, 90, 1):
        # select policy v3 set
        aug_policy = policies.policies_v2()

        # instantiate the policy container with the selected policy set
        policy_container = policies.PolicyContainer(aug_policy)

        # select a random policy from the policy set
        random_policy = policy_container.select_random_policy()
        # Apply the augmentation. Returns the augmented image and bounding boxes.
        # Image is a numpy array of the image
        # Bounding boxes is a list of list of bounding boxes in pixels (int).
        # e.g. [[x_min, y_min, x_man, y_max], [x_min, y_min, x_max, y_max]]
        # Labels are the class labels for the bounding boxes as an iterable of ints e.g. [1,0]
        img_aug, bbs_aug = policy_container.apply_augmentation(random_policy, img, bnd_box_list, class_name_number)
        # image_aug: numpy array of the augmented image
        # bbs_aug: numpy array of augmented bounding boxes in format: [[label, x_min, y_min, x_man, y_max],...]
        # print("Try " + str(num))
        # print(bbs_aug)
        # print(bbs_aug[:, 0])
        # print(bbs_aug[:, 1: 5])
        if bbs_aug.any():
            class_name_number2 = bbs_aug[:, 0]
            bnd_box_list2 = bbs_aug[:, 1: 5]

            class_name_list2 = []
            for class_name in class_name_number2:
                class_name_list2.append(class_name_dict.get(class_name))

            # print(class_name_list2)

            # filename_save = generate_random_string(5)
            filename_save = start
            start = generate_new_name(start)

            im = Image.fromarray(img_aug)
            im = im.convert('RGB')
            im.save("Dataset/pak cnic/augmented/" + filename_save + ".jpeg")

            label_file_write = 'Dataset/pak cnic/augmented/' + filename_save + '.xml'

            annotation_tag = ET.Element('annotation')

            # write filename to tree
            filename_tag = ET.SubElement(annotation_tag, 'filename')
            filename_tag.text = filename_save + ".jpeg"

            # write size to tree
            size_tag = ET.SubElement(annotation_tag, 'size')
            width_tag = ET.SubElement(size_tag, 'width')
            height_tag = ET.SubElement(size_tag, 'height')
            depth_tag = ET.SubElement(size_tag, 'depth')
            width_tag.text, height_tag.text, depth_tag.text = size

            # iterate through all bounding boxes and add to tree
            for i in range(len(class_name_list2)):
                object_tag = ET.SubElement(annotation_tag, 'object')

                # replace class_name_list with modified version
                name_tag = ET.SubElement(object_tag, 'name')
                name_tag.text = class_name_list2[i]

                # replace bnd_box_list with modified version
                bnd_box_tag = ET.SubElement(object_tag, 'bndbox')
                xmin_tag = ET.SubElement(bnd_box_tag, 'xmin')
                ymin_tag = ET.SubElement(bnd_box_tag, 'ymin')
                xmax_tag = ET.SubElement(bnd_box_tag, 'xmax')
                ymax_tag = ET.SubElement(bnd_box_tag, 'ymax')
                xmin_tag.text = str(bnd_box_list2[i][0])
                ymin_tag.text = str(bnd_box_list2[i][1])
                xmax_tag.text = str(bnd_box_list2[i][2])
                ymax_tag.text = str(bnd_box_list2[i][3])

            # create element tree to export as xml
            tree = ET.ElementTree(annotation_tag)
            tree.write(label_file_write, encoding="utf-8")
