import scipy.io
import numpy as np

image_size = 40 * 40
train_size = 60000
test_size = 10000

def read_data_from_file(file_index, num, train=True, one_of_n=True):
    if train:
        size = train_size
    else:
        size = test_size
    indexs = np.random.permutation(size)
    indexs = indexs[:num]
    images = read_images(file_index * size, size, train)
    labels = read_labels(file_index * size, size, train, one_of_n)

    return [images, labels]

def read_data_randomly(num, train=True, one_of_n=True):
    res_images = None
    res_labels = None
    per_num = int(num / 32)
    if train:
        size = train_size
    else:
        size = test_size
    for i in range(32):
        if i ==31:
            per_num = num - per_num * 31
        indexs = np.random.permutation(size)
        indexs = indexs[:per_num]
        images = read_images(i*size, size, train)
        labels = read_labels(i*size, size, train, one_of_n)
        # import IPython
        # IPython.embed()
        images = images[indexs]
        labels = labels[indexs]
        if res_images is None and res_labels is None:
            res_images = images
            res_labels = labels
        else:
            res_images = np.append(res_images, images)
            res_labels = np.append(res_labels, labels)
    res_images = res_images.reshape((-1, image_size))
    if one_of_n:
        res_labels = res_labels.reshape((-1, 10))
    return [res_images, res_labels]


def read_images(start, num, train=True, flat=True):
    res = None
    while num > 0:
        if train:
            file_index = int(start / 60000) + 1
            from_index = start % 60000
            data = scipy.io.loadmat("training_and_validation_batches/" + str(file_index) + ".mat")
            end_index = from_index + num
            if end_index >= 60000:
                start += train_size - from_index # need to read from start next loop
                num = end_index - 60000 # need to read num images next loop
                end_index = 60000 #
            else:
                start += num
                num = 0
        else:
            file_index = int(start / test_size) + 1
            from_index = start % test_size
            data = scipy.io.loadmat("test_batches/" + str(file_index) + ".mat")
            end_index = from_index + num
            if end_index >= test_size:
                start += test_size - from_index
                num = end_index - test_size
                end_index = test_size
            else:
                start += num
                num = 0
        data = data.get("affNISTdata")["image"]
        if res is None:
            res = data[0][0][:, from_index: end_index].transpose()
        else:
            res = np.append(res, data[0][0][:, from_index: end_index].transpose())
    # import IPython
    # IPython.embed()
    res = res.reshape((-1, image_size))
    return res

def read_labels(start, num, train=True, one_of_n=True):
    res = None
    while num > 0:
        if train:
            file_index = int(start / 60000) + 1
            from_index = start % 60000
            data = scipy.io.loadmat("training_and_validation_batches/" + str(file_index) + ".mat")
            end_index = from_index + num
            if end_index >= train_size:
                start += train_size - from_index
                num = end_index - train_size
                end_index = train_size
            else:
                start += num
                num = 0
        else:
            file_index = int(start / 10000) + 1
            from_index = start % 10000
            data = scipy.io.loadmat("test_batches/" + str(file_index) + ".mat")
            end_index = from_index + num
            if end_index >= test_size:
                start += test_size - from_index
                num = end_index - test_size
                end_index = test_size
            else:
                start += num
                num = 0
        if one_of_n:
            data = data.get("affNISTdata")["label_one_of_n"]
            if res is not None:
                res = np.append(res, data[0][0][:, from_index: end_index].transpose())
            else:
                res = data[0][0][:, from_index: end_index].transpose()
        else:
            data = data.get("affNISTdata")["label_int"]
            if res is not None:
                res = np.append(res, data[0][0][0][from_index: end_index].transpose())
            else:
                res = data[0][0][0][from_index: end_index].transpose()
    # import IPython
    # IPython.embed()
    if one_of_n:
        res = res.reshape(-1, 10)
    return res


if __name__ == "__main__":
    train_data = read_data(59999, 60001)
    test_data = read_data(59999, 2, train=False)

    train_labels = read_labels(59999, 60001)
    test_labels = read_labels(59999, 60001, train=False, one_of_n=False)

    print(train_data.shape)
    print(train_data)
    print(test_data.shape)
    print(test_data)
    print(train_labels.shape)
    print(train_labels)
    print(test_labels.shape)
    print(test_labels)

