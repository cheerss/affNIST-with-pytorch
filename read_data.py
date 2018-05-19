import scipy.io

image_height = 40
image_width = 40

def read_images(start, num, train=True, flat=True):
    if train:
        file_index = int(start / 60000) + 1
        from_index = start % 60000
        data = scipy.io.loadmat("training_and_validation_batches/" + str(file_index) + ".mat")
    else:
        file_index = int(start / 10000) + 1
        from_index = start % 10000
        data = scipy.io.loadmat("test_batches/" + str(file_index) + ".mat")
    data = data.get("affNISTdata")["image"]
    res = data[0][0][:, from_index: from_index+num].transpose()
    if not flat:
        res = res.reshape((-1, 1, image_height, image_width))
    else:
        res = res.reshape(-1, image_height * image_width)
    return res

def read_labels(start, num, train=True, one_of_n=True):
    if train:
        file_index = int(start / 60000) + 1
        from_index = start % 60000
        data = scipy.io.loadmat("training_and_validation_batches/" + str(file_index) + ".mat")
    else:
        file_index = int(start / 10000) + 1
        from_index = start % 10000
        data = scipy.io.loadmat("test_batches/" + str(file_index) + ".mat")
    if one_of_n:
        data = data.get("affNISTdata")["label_one_of_n"]
        return data[0][0][:, from_index: from_index+num].transpose()
    else:
        data = data.get("affNISTdata")["label_int"]
        return data[0][0][0][from_index: from_index+num].transpose()


if __name__ == "__main__":
    train_data = read_images(0, 100)
    test_data = read_images(0, 100, train=False)

    train_labels = read_labels(0, 100)
    test_labels = read_labels(0, 100, train=False, one_of_n=False)

    # print(train_data)
    # print(test_data)
    # print(train_labels)
    print(test_labels)

