import numpy as np
import time
from network.vae import VAE


def main(fn_train='network/dataset_bigwindow.npz', fn_vae='network/vae_bigwindow.net'):
    # Load dataset
    dataset = np.load(fn_train)
    x, y = dataset['x'], dataset['y']

    # Initialize network
    network = VAE()

    # Train network
    start = time.time()
    network.train_net(network, x, y)

    print('\nTime elapsed: {:.2f} seconds'.format(time.time() - start))

    # Save network
    network.save(fn_vae)


if __name__ == '__main__':
    main()
