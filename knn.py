import read_data
import torch
import collections
import time
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

gpu = True
train_batch = 1
all_trains = 1920000
all_tests = 100
top_k = 20
flat = True
test_batch = 1

def KNN(mat, k):
    mat_square = torch.mm(mat, mat.t())
    diag = torch.diagonal(mat_square)
    diag = diag.expand_as(mat_square)
    dist_mat = (diag + diag.t() - 2 * mat_square)
    dist_col = dist_mat[-1, :-1]
    # k+1 because the nearest must be itself
    print(dist_col.size())
    val, index = dist_col.topk(k, largest=False, sorted=True)
    return val, index


def main():
    accuracy = 0
    all = 0
    for i in range(test_batch):
        all_test_images = torch.Tensor(read_data.read_images(i * all_tests, all_tests, train=False, flat=flat))
        all_test_labels = torch.LongTensor(read_data.read_labels(i * all_tests, all_tests, train=False, one_of_n=False))

        if gpu:
            all_test_images = all_test_images.cuda()
            all_test_labels = all_test_labels.cuda()

        all_start = time.time()
        for test_index in range(all_test_images.size(0)):
            test_image = all_test_images[test_index].reshape((1, -1))
            test_label = all_test_labels[test_index]
            vals, truths = None, None
            for j in range(train_batch):
                start = time.time()
                all_train_images = torch.Tensor(read_data.read_images(j * all_trains, all_trains, train=True, flat=flat))
                all_train_labels = torch.LongTensor(read_data.read_labels(j * all_trains, all_trains, train=True, one_of_n=False))
                if gpu:
                    all_train_images = all_train_images.cuda()
                    all_train_labels = all_train_labels.cuda()
                end = time.time()
                print(all_train_images.size())
                print("read time for " + str(all_trains) + " data: " + str(end - start))
                batch_size = 10000
                for q in range(7):
                    train_images = all_train_images[q * batch_size: (q + 1) * batch_size]
                    train_labels = all_train_labels[q * batch_size: (q + 1) * batch_size]

                    mat = torch.cat((train_images, test_image), 0)
                    if vals is None:
                        vals, index = KNN(mat, top_k)
                        truths = train_labels[index]
                    else:
                        val, index = KNN(mat, top_k)
                        vals = torch.cat((vals, val), 0)
                        truths = torch.cat((truths, train_labels[index]), 0)
            nearest_k, nearest_indices = vals.topk(top_k, largest=False, sorted=True)
            neighbors_truth = truths[nearest_indices]
            predict = collections.Counter(neighbors_truth.tolist()).most_common(1)[0][0]
            if predict == test_label.tolist():
                accuracy += 1
            all += 1
            if(all % 10 == 0):
                print("tested cases: " + str(all))
            # print("predict: " + str(predict), end=" ")
            # print("truth: " + str(test_label.tolist()))
            # print(neighbors_truth.tolist())
            # print(test_label.tolist())
        all_end = time.time()

    accuracy /= all_tests * test_batch
    print("accuracy is " + str(accuracy) + " time is: " + str(all_end - all_start))


if __name__ == '__main__':
    main()