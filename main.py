import torch
import fnn
import cnn
import read_data
import time

iterations = 2000
batch_size = 1000
all_trains = 60000
all_tests = 10000
learning_rate = 0.008
gpu = True

def main():
    # net = fnn.FNN()
    net = cnn.CNN()
    print(net)
    start = time.time()
    for count in range(32):
        all_train_images = torch.Tensor(read_data.read_data(count * all_trains, all_trains, train=True))

        all_train_labels = torch.LongTensor(read_data.read_labels(count * all_trains, all_trains, train=True, one_of_n=False))
        print(all_train_images.shape)
        print(all_train_labels.shape)

        if gpu:
            net = net.cuda()
            all_train_images = all_train_images.cuda()
            all_train_labels = all_train_labels.cuda()

        for i in range(iterations):
            images_indexs = torch.randperm(all_trains)
            images_indexs = images_indexs[0:batch_size]

            if gpu:
                images_indexs = images_indexs.cuda()

            images = torch.autograd.Variable(all_train_images[images_indexs])
            labels = torch.autograd.Variable(all_train_labels[images_indexs])
            # print(images.shape)
            # print(labels.shape)

            output = net(images)

            criterion = torch.nn.NLLLoss()
            # print(output.shape)
            loss = criterion(output, labels)
            if i % 100 == 0:
                print(loss)

            net.zero_grad()
            loss.backward()
            for f in net.parameters():
                f.data.sub_(f.grad.data * learning_rate)

    end = time.time()
    print("training time: " + str(end-start))
        # output = net(torch.autograd.Variable(all_train_images))
        # output = torch.max(output, dim=1)[1]
        # # print(type(output))
        # # print(output.shape)
        # res = output.data - all_train_labels
        # error_rate = torch.nonzero(res).size()[0]
        # # import IPython
        # # IPython.embed()
        # error_rate /= 60000
        # print("train error rate " + str(count) + " : " + str(error_rate))
        # # print(output)
        # # print(all_train_labels[:15])

    train_error_rate = 0
    for i in range(32):
        all_train_images = torch.Tensor(read_data.read_data(i * all_trains, all_trains, train=True))
        all_train_labels = torch.LongTensor(read_data.read_labels(i * all_trains, all_trains, train=True, one_of_n=False))
        if gpu:
            all_train_images = all_train_images.cuda()
            all_train_labels = all_train_labels.cuda()
        output = net(torch.autograd.Variable(all_train_images))
        output = torch.max(output, dim=1)[1]
        res = output.data - all_train_labels
        train_error_rate += torch.nonzero(res).size()[0]
    train_error_rate /= 32 * all_trains
    print("train accuracy: " + str(1-train_error_rate))

    test_error_rate = 0
    for i in range(32):
        all_test_images = torch.Tensor(read_data.read_data(i * 10000, 10000, train=False))
        all_test_labels = torch.LongTensor(read_data.read_labels(i * 10000, 10000, train=False, one_of_n=False))
        if gpu:
            all_test_images = all_test_images.cuda()
            all_test_labels = all_test_labels.cuda()
        output = net(torch.autograd.Variable(all_test_images))
        output = torch.max(output, dim=1)[1]
        res = output.data - all_test_labels
        test_error_rate += torch.nonzero(res).size()[0]
    test_error_rate /= 32 * all_tests
    print("test accuracy : " + str(1-test_error_rate))

if __name__ == "__main__":
    main()
