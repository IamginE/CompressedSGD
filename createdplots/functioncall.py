plot('batch_loss_hist', 
     files=['../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n2_llin1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n2_llin1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n2_llin1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n2_llin1_EvalInterval50_Seed1917/histories.pkl'],
     keys=[(0.001, 1), (0.0005, 1),
           (0.001, 1.0), (0.001, 0.9),
           (0.0005, 1.0), (0.0005, 0.9)
           ],
     names=['signSGD lr=0.001', 'signSGD lr=0.0005',
            'compressedSGD 2bins lr=0.001, decay=1.0', 'compressedSGD 2bins lr=0.001, decay=0.9',
            'compressedSGD 2bins lr=0.0005, decay=1.0', 'compressedSGD 2bins lr=0.0005, decay=0.9'],
     out='cSGD_2bits_sign.png',
     linspace_mul=50)

plot('batch_loss_hist', 
     files=['../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917/histories.pkl'],
     keys=[(0.001, 1), (0.0005, 1),
           (0.001, 1.0), (0.0005, 1.0),
           (0.00025, 1.0)
           ],
     names=['signSGD lr=0.001', 'signSGD lr=0.0005',
            'compressedSGD 3bins lr=0.001, decay=1.0', 'compressedSGD 3bins lr=0.0005, decay=1.0',
            'compressedSGD 3bins lr=0.00025, decay=1.0'],
     out='cSGD_3bits_decay10_sign.png',
     linspace_mul=50)

plot('batch_loss_hist', 
     files=['../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917/histories.pkl'],
     keys=[(0.001, 1), (0.0005, 1),
           (0.001, 0.9), (0.0005, 0.9),
           (0.00025, 0.9)
           ],
     names=['signSGD lr=0.001', 'signSGD lr=0.0005',
            'compressedSGD 3bins lr=0.001, decay=0.9', 'compressedSGD 3bins lr=0.0005, decay=0.9',
            'compressedSGD 3bins lr=0.00025, decay=0.9'],
     out='cSGD_3bits_decay09_sign.png',
     linspace_mul=50)

plot('batch_loss_hist', 
     files=['../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_2/histories.pkl'],
     keys=[(0.001, 1), (0.0005, 1),
           (0.001, 0.8), (0.0005, 0.8),
           (0.00025, 0.8)
           ],
     names=['signSGD lr=0.001', 'signSGD lr=0.0005',
            'compressedSGD 3bins lr=0.001, decay=0.8', 'compressedSGD 3bins lr=0.0005, decay=0.8',
            'compressedSGD 3bins lr=0.00025, decay=0.8'],
     out='cSGD_3bits_decay08_sign.png',
     linspace_mul=50)

plot('batch_loss_hist', 
     files=['../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_2/histories.pkl'],
     keys=[(0.001, 1), (0.0005, 1),
           (0.001, 0.7), (0.0005, 0.7),
           (0.00025, 0.7)
           ],
     names=['signSGD lr=0.001', 'signSGD lr=0.0005',
            'compressedSGD 3bins lr=0.001, decay=0.7', 'compressedSGD 3bins lr=0.0005, decay=0.7',
            'compressedSGD 3bins lr=0.00025, decay=0.7'],
     out='cSGD_3bits_decay07_sign.png',
     linspace_mul=50)

plot('batch_loss_hist', 
     files=['../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_2/histories.pkl'],
     keys=[(0.002, 1), (0.0015, 1),
           (0.001, 0.7), (0.0005, 0.7),
           (0.00025, 0.7)
           ],
     names=['signSGD lr=0.002', 'signSGD lr=0.0015',
            'compressedSGD 3bins lr=0.001, decay=0.7', 'compressedSGD 3bins lr=0.0005, decay=0.7',
            'compressedSGD 3bins lr=0.00025, decay=0.7'],
     out='cSGD_3bits_decay07_sign_2.png',
     linspace_mul=50)

plot('batch_loss_hist', 
     files=['../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/compressed_sgd_B64_n2_llin1_EvalInterval50_Seed1917_3/histories.pkl',
      '../logs/compressed_sgd_B64_n2_llin1_EvalInterval50_Seed1917_3/histories.pkl',
      '../logs/compressed_sgd_B64_n2_llin1_EvalInterval50_Seed1917_3/histories.pkl',
      '../logs/compressed_sgd_B64_n2_llin1_EvalInterval50_Seed1917_3/histories.pkl'],
     keys=[(0.002, 1), (0.0015, 1),
           (0.002, 0.8), (0.0015, 0.8),
           (0.002, 0.7), (0.0015, 0.8)
           ],
     names=['signSGD lr=0.002', 'signSGD lr=0.0015',
            'compressedSGD 2bits lr=0.002, decay=0.8', 'compressedSGD 2bins lr=0.0015, decay=0.8',
            'compressedSGD 2bits lr=0.002, decay=0.7', 'compressedSGD 2bins lr=0.0015, decay=0.7'],
     out='cSGD_2bits_decay08_07_sign_2.png',
     linspace_mul=50)

plot('batch_loss_hist', 
     files=['../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/compressed_sgd_B64_n2_llin1_EvalInterval50_Seed1917_3/histories.pkl',
      '../logs/compressed_sgd_B64_n2_llin1_EvalInterval50_Seed1917_3/histories.pkl',
      '../logs/compressed_sgd_B64_n2_llin1_EvalInterval50_Seed1917_3/histories.pkl',
      '../logs/compressed_sgd_B64_n2_llin1_EvalInterval50_Seed1917_3/histories.pkl'],
     keys=[(0.002, 1), (0.0015, 1),
           (0.002, 1.0), (0.0015, 1.0),
           (0.002, 0.9), (0.0015, 0.9)
           ],
     names=['signSGD lr=0.002', 'signSGD lr=0.0015',
            'compressedSGD 2bits lr=0.002, decay=1.0', 'compressedSGD 2bits lr=0.0015, decay=1.0',
            'compressedSGD 2bits lr=0.002, decay=0.9', 'compressedSGD 2bits lr=0.0015, decay=0.9'],
     out='cSGD_2bits_decay10_09_sign_2.png',
     linspace_mul=50)

plot('batch_loss_hist', 
     files=['./logs/sign_sgd_B64_e1_EvalInterval50_Seed1917_2/histories.pkl',
      './logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      './logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917/histories.pkl',
      './logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917/histories.pkl',
      './logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_2/histories.pkl',
      './logs/compressed_sgd_B64_n3_llin1_EvalInterval50_Seed1917_2/histories.pkl'],
     keys=[(0.0015, 1), (0.001, 1),
           (0.001, 1.0), (0.0005, 0.9),
           (0.001, 0.8), (0.0005, 0.7)
           ],
     names=['signSGD lr=0.0015', 'signSGD lr=0.001',
            'compressedSGD 3bits lr=0.001, decay=1.0', 'compressedSGD 3bits lr=0.0005, decay=0.9',
            'compressedSGD 3bits lr=0.001, decay=0.8', 'compressedSGD 3bits lr=0.0005, decay=0.7'],
     out='cSGD_3bits_decay_best_sign_2.png',
     linspace_mul=50)

plot('batch_loss_hist', 
     files=['../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n4_llin1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n4_llin1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n4_llin1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/compressed_sgd_B64_n4_llin1_EvalInterval50_Seed1917/histories.pkl'],
     keys=[(0.0015, 1), (0.001, 1),
           (0.0005, 0.8), (0.0005, 0.7),
           (0.00025, 0.9), (0.00025, 0.8)
           ],
     names=['signSGD lr=0.0015', 'signSGD lr=0.001',
            'compressedSGD 4bits lr=0.0005, decay=0.8', 'compressedSGD 4bits lr=0.0005, decay=0.7',
            'compressedSGD 4bits lr=0.00025, decay=0.9', 'compressedSGD 4bits lr=0.00025, decay=0.8'],
     out='cSGD_4bits_decay10-07_sign.png',
     linspace_mul=50)

plot('batch_loss_hist', 
     files=['../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917_2/histories.pkl',
      '../logs/sign_sgd_B64_e1_EvalInterval50_Seed1917/histories.pkl',
      '../logs/sgd_B64_e1_EvalInterval50_Seed1917_2/histories.pkl'
      ],
     keys=[(0.0015, 1), (0.001, 1),
           (0.5, 1),
           ],
     names=['signSGD lr=0.0015', 'signSGD lr=0.001',
            'SGD lr=0.5'],
     out='SGD_signSGD.png',
     linspace_mul=50)


#Num workers
plot('avg_epoch_loss_hist', 
        files=[
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e2_EvalInterval100_Seed1917_Workers1/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e2_EvalInterval100_Seed1917_Workers1/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e4_EvalInterval100_Seed1917_Workers2/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e4_EvalInterval100_Seed1917_Workers2/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e8_EvalInterval100_Seed1917_Workers4/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e8_EvalInterval100_Seed1917_Workers4/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e16_EvalInterval100_Seed1917_Workers8/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e16_EvalInterval100_Seed1917_Workers8/histories.pkl',
        ],
        keys=[
            (0.0001, 0.3), (0.0001, 0.7),
            (0.0001, 0.3), (0.0001, 0.7),
            (0.0001, 0.3), (0.0001, 0.7),
            (0.0001, 0.3), (0.0001, 0.7),
            (0.0001, 0.3), (0.0001, 0.7),
            (0.0001, 0.3), (0.0001, 0.7),
            ],
        names=[
            'compressedSGD 3bits lr=0.0001, decay=0.3 workers=1', 'compressedSGD 3bits lr=0.0001, decay=0.7 workers=1',
            'compressedSGD 3bits lr=0.0001, decay=0.3 workers=2', 'compressedSGD 3bits lr=0.0001, decay=0.7 workers=2',
            'compressedSGD 3bits lr=0.0001, decay=0.3 workers=4', 'compressedSGD 3bits lr=0.0001, decay=0.7 workers=4',
            'compressedSGD 3bits lr=0.0001, decay=0.3 workers=8', 'compressedSGD 3bits lr=0.0001, decay=0.7 workers=8',
            ],
        out='cSGD_3bits_num_workers_epoch_loss.png',
        xaxis='Epochs',
        yaxis='Loss',
        title='Effect of Worker Count',
        linspace_mul=1)

plot('batch_loss_hist', 
        files=[
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e2_EvalInterval100_Seed1917_Workers1/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e2_EvalInterval100_Seed1917_Workers1/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e4_EvalInterval100_Seed1917_Workers2/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e4_EvalInterval100_Seed1917_Workers2/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e8_EvalInterval100_Seed1917_Workers4/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e8_EvalInterval100_Seed1917_Workers4/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e16_EvalInterval100_Seed1917_Workers8/histories.pkl',
            './used_data/num_workers/compressed_sgd_vote_B64_n3_blin_e16_EvalInterval100_Seed1917_Workers8/histories.pkl',
        ],
        keys=[
            (0.0001, 0.3), (0.0001, 0.7),
            (0.0001, 0.3), (0.0001, 0.7),
            (0.0001, 0.3), (0.0001, 0.7),
            (0.0001, 0.3), (0.0001, 0.7),
            (0.0001, 0.3), (0.0001, 0.7),
            (0.0001, 0.3), (0.0001, 0.7),
            ],
        names=[
            'compressedSGD 3bits lr=0.0001, decay=0.3 workers=1', 'compressedSGD 3bits lr=0.0001, decay=0.7 workers=1',
            'compressedSGD 3bits lr=0.0001, decay=0.3 workers=2', 'compressedSGD 3bits lr=0.0001, decay=0.7 workers=2',
            'compressedSGD 3bits lr=0.0001, decay=0.3 workers=4', 'compressedSGD 3bits lr=0.0001, decay=0.7 workers=4',
            'compressedSGD 3bits lr=0.0001, decay=0.3 workers=8', 'compressedSGD 3bits lr=0.0001, decay=0.7 workers=8',
            ],
        out='cSGD_3bits_num_workers_batch_loss.png',
        xaxis='Epochs',
        yaxis='Loss',
        title='Effect of Worker Count',
        linspace_mul=100)

from matplotlib import pyplot as plt
from PIL import Image

fig, axs = plt.subplots(2, 2, figsize=(20, 20), squeeze=False)
axs[0,0].imshow(Image.open('./used_data/plots/epochs_decay_lr_all_3bits.png'))
axs[0,0].axis('off')
axs[0,1].imshow(Image.open('./used_data/plots/epochs_decay_lr_focused_3bits.png'))
axs[0,1].axis('off')
axs[1,0].imshow(Image.open('./used_data/plots/epochs_decay_lr_all_5bits.png'))
axs[1,0].axis('off')
axs[1,1].imshow(Image.open('./used_data/plots/epochs_decay_lr_focused_5bits.png'))
axs[1,1].axis('off')
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('LRs_Decays.png', bbox_inches='tight')