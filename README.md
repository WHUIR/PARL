## Requirements

Tensorflow 1.2

Python 2.7

Numpy

Scipy

## Data Preparation
To run PARL, 7 files are required: 

1.**Training Rating records: file_name=TrainInteraction.out**  
each training sample is a sequence as:  
UserId\tItemId\tRating\tDate  
Example: 0\t3\t5.0\t1393545600  

2.**Validate Rating records: file_name=ValidateInteraction.out**  
The format is the same as the training data format.  

3.**Testing Rating records: file_name=TestInteraction.out**  
The format is the same as the training data format.  

4.**Word2Id diction: file_name=WordDict.out**  
Each line follows the format as:  
Word\tWord_Id  
Example: love\t0  

5.**User Review Document: file_name=UserReviews.out**  
each line is the format as:  
UserId\tWord1 Word2 Word3 …  
Example:0\tI love to eat hamburger …  

6.**Item Review Document: file_name=ItemReviews.out**  
The format is the same as the user review doc format.  

7.**User Auxiliary Review Document: file_name=UserAuxiliaryReviews.out**  
The format is the same as the user review doc format.  

### ***Note that: all files need to be located in the same directory***

## Configurations
num_factor: the latent dimension of the representation learned from the review documents;

num_filters: the number of filters of CNN network;

cnn_windows_size: the length of the sliding window of CNN;

learn_rate: learning rate;

batch_size: batch size;

epochs: number of training epoch;

max_len: the maximum doc length;

gama: the coefficient to control the constraint between tu & tax;

drop_rate: the keep probability of the drop_out strategy;
