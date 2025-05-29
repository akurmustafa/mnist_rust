# download data
You can download the dataset from the following kaggle link (unfortunately requires login)
[kaggle mnist csv](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data)

Please place the data under the folder `./data/mnist/` in the `.git` directory.

# Execute code
Following command does a single training and prints the accuracy on the test data after training.
`cargo run -- --mnist './data/mnist' --batch-size 64 --learning-rate 0.001 --print-every 50`

If you want to limit iteration in the training, you can do so with 
`cargo run -- --mnist './data/mnist' --batch-size 64 --learning-rate 0.001 --print-every 50 --n_iter=100`