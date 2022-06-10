import optimizers
import models
import data_loaders
from trainer import Trainer
from config import get_configs

def main():
    args = get_configs()

    model = models.__dict__[args.model](
            num_classes=args.num_classes,
            pretrianed=args.pretrained
            )

    optimizer = optimizers.__dict__[args.optimizer](
            params=model.parameters(),
            lr=args.lr,
            lr_decay=args.lr_decay,
            decay_min=args.decay_min,
            decay_max=args.decay_max,
            num_bits=args.num_bits, 
            )

    train_loader = data_loaders.__dict__[args.dataset](
            root=args.dataset_root,
            split='train',
            batch_size=args.batch_size,
            num_workers=args.num_workers
            )
    
    trainer = Trainer(model=model,
                      data_loader=train_loader,
                      optimizer=optimizer,
                      args=args
                      )
    history = trainer.train(args.epochs)


if __name__=='__main__':
    main()