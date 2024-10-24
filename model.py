import os
import torch
import torchvision

import configs
import utils
from alexnet import AlexNet
from data import ThumbnailDataset, BlackandWhite



class VEAModel:
    def __init__(self):
        self.model = AlexNet(num_classes=2)
        dataset = ThumbnailDataset
        self.train_dataloader = torch.utils.data.DataLoader(dataset=dataset(mode="train"), batch_size=configs.BATCH_SIZE, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=dataset(mode="test"), batch_size=2, shuffle=False)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=configs.LEARNING_RATE)
        self.loss_fn = torch.nn.MSELoss()

    def train(self):
        min_loss = float("inf")
        self.model.train()
        for epoch in range(configs.NUM_EPOCHS):

            # Reset epoch loss
            loss_epoch = []

            for batch, (images, targets) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                #print(images.shape, targets.shape)
                outputs = self.model(images)
                
                # Convert classification logits to score
                targets = torch.stack([torch.tensor([0,1]) if t > 0 else torch.tensor([1,0]) for t in targets]).float()
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                scores = outputs

                # Acc
                a = torch.argmax(targets, dim=1)
                b = torch.argmax(scores, dim=1)
                acc = torch.sum((a==b).float())/len(a)

                # Loss
                loss = self.loss_fn(scores, targets)
                loss_epoch.append(loss)
                print("INFO:", "Epoch", epoch, "Batch", batch, "Loss", loss.item(), "Acc", acc.item())

                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Validation
            self.val()

            # Save model
            loss_epoch = torch.mean(torch.tensor(loss_epoch))
            if loss_epoch < min_loss:
                print("INFO: Save Model")
                min_loss = loss_epoch
                torch.save(self.model, os.path.join(configs.ROOT, "./best_model.pth"))
            
    def val(self):
        mean_acc = []
        with torch.no_grad():
            for batch, (images, targets) in enumerate(self.test_dataloader):
                outputs = self.model(images)

                # Convert classification logits to score
                targets = torch.stack([torch.tensor([0,1]) if t > 0 else torch.tensor([1,0]) for t in targets]).float()
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                scores = outputs

                # Acc
                a = torch.argmax(targets, dim=1)
                b = torch.argmax(scores, dim=1)
                acc = torch.sum((a==b).float())/len(a)
                mean_acc.append(acc)
        
        mean_acc = sum(mean_acc)/len(mean_acc)
        print("INFO:", "Validation Acc", mean_acc.item())

    def test(self):
        # Load model
        self.model = torch.load(os.path.join(configs.ROOT, "./best_model.pth"), weights_only=False)

        # Inference
        with torch.no_grad():
            for images, targets in self.test_dataloader:
                outputs = self.model(images)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                outputs = torch.argmax(outputs, dim=1)

                # Preview
                utils.save_image(images[0], os.path.join(configs.ROOT, "./preview.jpg"))
                print("INFO:", "Output", outputs, "Target", targets)
                input("Press ENTER for next image!")

    def inference(self, x):
        # Load model
        self.model = torch.load(os.path.join(configs.ROOT, "./best_model.pth"), weights_only=False)

        # Inference 
        outputs = self.model(x)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Calculate scores
        scores = [(s[1]-s[0]) for s in outputs]
        return scores


if __name__ == "__main__":
    model = VEAModel()
    if configs.MODE == "train": model.train()
    if configs.MODE == "test": model.test()