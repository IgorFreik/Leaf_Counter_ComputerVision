import torch
import torch.optim as optim
from data_prep import prepare_data
from resnet_unet import LeafCounter
from train_test import train, evaluate
from gradio_interface import show_web_interface


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dl, test_dl = prepare_data()
    model = LeafCounter().to(device)
    train(model=model,
          epochs=10,
          data_tr=train_dl,
          data_val=test_dl,
          device=device,
          opt_count=optim.Adam(model.counter.parameters(), lr=0.01),
          opt_segm=optim.Adam(model.segmenter.parameters(), lr=0.01))
    evaluate(model, test_dl, device)
    show_web_interface(model)
