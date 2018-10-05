'''Train unsuperwised entity grounding by attention+pixel classification mechanism.'''
from __future__ import print_function

import random
import pickle
from parser import *
import matplotlib.pyplot as plt
from models.Model7 import Model7
from lib.configure.net_util import *
from torchvision import transforms
from tensorboardX import SummaryWriter
from lib.dataset.coco_dataset import CocoCaptions


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def l2_regulariza_loss(map):
    # return torch.mean(map.view(map.shape[0], map.shape[-2], map.shape[-1]))
    mean = torch.mean(map.view(map.shape[0], map.shape[-2], map.shape[-1]))
    return mean


def load_dictionary(name):
    with open('./others/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Randomly pick a label from multi one-hot label
def random_pick(one_hot):
    # return a randomly selected label
    label = torch.zeros(one_hot.shape[0])
    one_hot_return = torch.zeros_like(one_hot)

    for i in range(one_hot.shape[0]):
        # all labels to save all the labels
        all_labels = []
        count = 0
        for j in range(one_hot.shape[1]):
            if one_hot[i][j] == 1.:
                all_labels.append(count)
            count += 1
        # randomly picking one label
        if len(all_labels) != 0:
            label[i] = random.choice(all_labels)
        else:
            label[i] = 2
        one_hot_return[i][int(label[i])] = 1
    return label, one_hot_return


# Multi-Pixel embedding learning for multi-category picking
def top_k_emb(visual_emb, model, label, single_attribute_label, K=100):
    # Given pixel-wise features, select top-k pixels with highest category prob out for
    # multi-cross entropy learning
    # Visual-features:   (batch, emb #, pixel #)
    # Returning prob: (batch #, top-K, class_prob)
    # Returning feat: (batch #, top-K, feature_size)
    visual_emb = visual_emb.view((visual_emb.shape[0], visual_emb.shape[1], visual_emb.shape[2]*visual_emb.shape[3]))

    # i: batch number
    for i in range(visual_emb.shape[0]):
        sorting = np.zeros((visual_emb.shape[2]))
        # j: pixel numbers in feature maps
        for j in range(visual_emb.shape[2]):
            # extracting pixel features and reshape
            emb = visual_emb[i, :, j]
            # emb = F.relu(model.fc_p5(emb.contiguous().view(1, -1)))
            emb_ = (emb.contiguous().view(1, -1))
            output = model.attr_res_net.fc(emb_)
            prob = opts.criterion[0](output, single_attribute_label[i])
            opts.prob_set[j] = output[0]
            opts.features_set[j] = emb
            sorting[j] = prob.data.cpu().numpy()[0]

        # Arg-sort the probability (and inverse the order)
        sorting = np.argsort(sorting)[0:K]

        # index: number of top-K
        for index in range(K):
            opts.return_prob[i, index] = opts.prob_set[int(sorting[index])]
            opts.return_feat[i, index] = opts.features_set[int(sorting[index])]
    return opts.return_feat


def train_net(net, opts):

    print('training at epoch {}'.format(opts.epoch+1))

    if opts.use_gpu:
        net.cuda()

    net.train(True)
    train_loss = 0
    total_time = 0
    batch_idx = 0
    optimizer = opts.current_optimizer
    # back_bone_optimizer = opts.backbone_optimizer
    end_time = time.time()
    train_back_bone = True
    fig = plt.figure()

    # category:         semantic labels for single selected label
    # s_entity_one_hot: randomly selected entity one-hot
    # s_entity_label:   randomly selected entity label
    # att_emb:          word2vec embedding for attributes
    # att_label:        attributes pairs for margin loss learning
    # attr_one_hot:     all attributes one-hot
    # textual_emb:      phrase embedding
    # phrase/line:      phrases/lines in NLP format
    # mask:             ground truth annotations for object
    for batch_idx, (images, attr_one_hot, entity_one_hot) in enumerate(data_loader):

        # model.visual_net.config.IMAGES_PER_GPU = images.size(0)
        images = Variable(images).cuda()

        # Randomly pick one attribute per iteration
        single_attribute_label, single_attribute_one_hot = random_pick(attr_one_hot)
        attr_one_hot = Variable(single_attribute_one_hot).cuda().float()
        single_attribute_label = Variable(single_attribute_label).cuda().long()

        # Create embeddings input
        embeddings = Variable(torch.zeros(attr_one_hot.shape[0], 300))
        for index, item in enumerate(single_attribute_label):
            i = opts.entity_att[item.data.cpu().numpy()[0]]
            embeddings[index] = Variable(torch.from_numpy(opts.embeddings_index[i])).cuda()

        # Feed in network
        y, attr_map, att_conv_feature = net(images, single_attribute_label, embeddings)

        loss = y

        if train_back_bone:
            optimizer.zero_grad()
            train_loss += loss.data[0]
            loss.backward()
            optimizer.step()

        # Display the generated att_map and instant loss
        if batch_idx % 1 == 0:
            plt.ion()
            plt.show()
            random = randint(0, opts.batch_size - 1)
            if batch_idx % 1 == 0:
                # Print out the attribute labels
                # plt.suptitle(opts.entity_att[int(single_attribute_label[random])])
                plt.subplot(141)
                vis = torch.nn.functional.sigmoid((model.attr_res_net.fc.weight[0].view(-1, 1, 1)
                                                   * att_conv_feature[random]).sum(0)).cpu().data.numpy()
                plt.imshow(vis)

                plt.subplot(142)
                vis = torch.nn.functional.sigmoid((model.attr_res_net.fc.weight[1].view(-1, 1, 1)
                                                   * att_conv_feature[random]).sum(0)).cpu().data.numpy()
                plt.imshow(vis)

                plt.subplot(143)
                plt.imshow(attr_map[random, 0].data.cpu().numpy())

                plt.subplot(144)
                plt.imshow(images[random].permute(1, 2, 0).float().data.cpu())
                plt.pause(0.001)
                writer.add_scalar('Cross Entropy Loss', train_loss / (batch_idx+1), opts.iter_n)
                opts.iter_n += 1

        print('Overall Loss: %.8f'
              % (train_loss/(batch_idx+1)))

        total_time += (time.time() - end_time)
        end_time = time.time()
        batch_idx += 1

        opts.train_batch_logger.log({
            'epoch': (opts.epoch+1),
            'batch': batch_idx+1,
            'loss': train_loss / (batch_idx+1),
        })

        if batch_idx % 100 == 0:
            print('100 batch.')
            # Save checkpoint.
            net_states = {
                'state_dict': net.state_dict(),
                'epoch': opts.epoch + 1,
                'loss': opts.train_losses,
                'optimizer': opts.current_optimizer.state_dict()
            }
            epo_batch = str(opts.epoch) + '-' + str(batch_idx)
            save_file_path = os.path.join(opts.checkpoint_path,
                                          'Model7_exp1_{}.pth'.format(epo_batch))
            torch.save(net_states, save_file_path)
            opts.lr /= 2
            opts.regularization /= 2
            params = filter(lambda p: p.requires_grad, model.parameters())
            opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    train_loss /= (batch_idx + 1)

    opts.train_epoch_logger.log({
        'epoch': (opts.epoch+1),
        'loss': train_loss,
        'time': total_time,
    })

    opts.train_losses.append(train_loss)

    # Save checkpoint.
    net_states = {
        'state_dict': net.state_dict(),
        'epoch': opts.epoch + 1,
        'loss': opts.train_losses,
        'optimizer': opts.current_optimizer.state_dict()
    }

    if opts.epoch % opts.checkpoint_epoch == 0:
        save_file_path = os.path.join(opts.checkpoint_path, 'Model7_exp1_{}.pth'.format(opts.epoch))
        torch.save(net_states, save_file_path)

    print('Batch Loss: %.8f, elapsed time: %3.f seconds.' % (train_loss, total_time))


if __name__ == '__main__':

    opts = parse_opts()
    writer = SummaryWriter()

    if opts.gpu_id >= 0:
        torch.cuda.set_device(opts.gpu_id)
        opts.multi_gpu = False

    torch.manual_seed(opts.seed)
    if opts.use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed(opts.seed)

    # Loading Data
    print("Preparing Flickr data set...")
    opts.k = 600
    opts.ite = 0
    opts.regularization = 0.1
    size = (1024, 1024)
    feat_size = (64, 64)
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    data_set = CocoCaptions(opts.img_path, opts.annotation, transform)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=opts.batch_size, shuffle=True)

    # Load dictionary
    list_file = open(opts.dictionary, 'r')
    entity_att = []
    for i in list_file.readlines():
        entity_att.append(i.replace('\n', ''))
    opts.entity_att = entity_att

    # Load semantic embeddings
    embeddings_index = load_dictionary('dictionary_emb')
    print('Dictionary loaded.')
    opts.embeddings_index = embeddings_index

    if not os.path.exists(opts.result_path):
        os.mkdir(opts.result_path)

    opts.train_epoch_logger = Logger(os.path.join(opts.result_path, 'train.log'),
                                     ['epoch', 'time', 'loss'])
    opts.train_batch_logger = Logger(os.path.join(opts.result_path, 'train_batch.log'),
                                     ['epoch', 'batch', 'loss'])
    opts.test_epoch_logger = Logger(os.path.join(opts.result_path, 'test.log'),
                                    ['epoch', 'time', 'loss'])

    # Model
    print('==> Building model...')
    model = Model7(opts)

    # Load Back bone Module
    if opts.resume:
        state_dict = torch.load(opts.resume)['state_dict']
        new_params = model.state_dict()
        new_params.update(state_dict)
        # Remove the extra keys
        model_keys = model.state_dict().keys()
        for name, param in list(new_params.items()):
            if name not in model_keys:
                del new_params[name]
        model.load_state_dict(new_params)
    start_epoch = 0
    print('==> model built.')
    opts.criterion = [torch.nn.CrossEntropyLoss(), torch.nn.MSELoss()]

    # Training
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in parameters])
    print(params, 'trainable parameters in the network.')
    set_parameters(opts)
    opts.iter_n = 0

    for epoch in range(start_epoch, start_epoch+opts.n_epoch):
        opts.epoch = epoch
        if epoch is 0:
            params = filter(lambda p: p.requires_grad, model.parameters())
            opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

        elif (epoch % opts.lr_adjust_epoch) == 0 and epoch is not 0:
            opts.lr /= 5
            params = filter(lambda p: p.requires_grad, model.parameters())
            opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

        train_net(model, opts)

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
