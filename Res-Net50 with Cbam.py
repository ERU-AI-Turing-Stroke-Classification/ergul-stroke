import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torchsummary import summary
from torchvision import transforms

from models.resnet_cbam import ResNet50_CBAM  # CBAM'li model
from runtime_args import args
from load_dataset import LoadDataset
from plot import plot_loss_acc
from helpers import calculate_accuracy

device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")

if not os.path.exists(args.graphs_folder):
    os.mkdir(args.graphs_folder)

model_save_folder = 'resnet_cbam/' if args.use_cbam else 'resnet/'
if not os.path.exists(model_save_folder):
    os.mkdir(model_save_folder)

def train(args):

    model = ResNet50_CBAM(image_depth=args.img_depth, num_classes=args.num_classes, use_cbam=args.use_cbam).to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    summary(model, (1, 224, 224))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=True, transform=transform)
    test_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=False, transform=transform)

    train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    training_loss_list = []
    training_acc_list = []
    testing_loss_list = []
    testing_acc_list = []

    best_accuracy = 0

    for epoch_idx in range(args.epoch):
        model.train()

        epoch_loss = []
        epoch_accuracy = []

        for sample in tqdm(train_generator):
            batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)

            optimizer.zero_grad()
            _, net_output = model(batch_x)
            total_loss = criterion(net_output, batch_y)

            total_loss.backward()
            optimizer.step()

            batch_accuracy = calculate_accuracy(predicted=net_output, target=batch_y)
            epoch_loss.append(total_loss.item())
            epoch_accuracy.append(batch_accuracy)

        curr_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
        curr_loss = sum(epoch_loss) / len(epoch_loss)

        training_loss_list.append(curr_loss)
        training_acc_list.append(curr_accuracy)

        print(f"Epoch {epoch_idx}: Training Loss = {curr_loss}, Training Accuracy = {curr_accuracy}")

        model.eval()
        epoch_loss = []
        epoch_accuracy = []

        with torch.no_grad():
            for sample in tqdm(test_generator):
                batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)
                _, net_output = model(batch_x)
                total_loss = criterion(net_output, batch_y)

                batch_accuracy = calculate_accuracy(predicted=net_output, target=batch_y)
                epoch_loss.append(total_loss.item())
                epoch_accuracy.append(batch_accuracy)

            curr_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
            curr_loss = sum(epoch_loss) / len(epoch_loss)

            testing_loss_list.append(curr_loss)
            testing_acc_list.append(curr_accuracy)

        print(f"Epoch {epoch_idx}: Testing Loss = {curr_loss}, Testing Accuracy = {curr_accuracy}")

        plot_loss_acc(path=args.graphs_folder, num_epoch=epoch_idx, train_accuracies=training_acc_list, train_losses=training_loss_list,
                      test_accuracies=testing_acc_list, test_losses=testing_loss_list)

        if epoch_idx % 5 == 0:
            lr_decay.step()
            print(f"Yeni Öğrenme Oranı: {optimizer.param_groups[0]['lr']}")

        if best_accuracy < curr_accuracy:
            torch.save(model.state_dict(), f"{model_save_folder}best_model.pth")
            best_accuracy = curr_accuracy
            print("Yeni en iyi model kaydedildi!")

        print("\n" + "-"*80 + "\n")

if __name__ == '__main__':
    train(args)
