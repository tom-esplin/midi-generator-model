# midi-generator-model

My deep learning final project, uses transformers and lstms to extend midi(Musical Instruments Digital Interface or note data) sequences as input to my synthesizer.

## Plan

My goal for the final project is to create a model that given a sequence of MIDI(musical
instrument digital interface) notes played can extend it to a longer sequence of notes in an auto
regressive fashion similar to an LLM. These sequences of notes should make sense musically
and maintain patterns throughout. The hope is this will give me inspiration to write and create
more music.
I plan on trying two different approaches. First, I want to try a transformer architecture.
This is the most state of art solution for symbolic music generation today but it is the most
difficult to train. I read that using an adam optimizer and a training schedule can help but
tweaking and messing with the hyperparameters is essential in order to successfully run the
model. It is recommended that all training steps are extensively unit tested otherwise the model
may fail to train properly. Additionally, the transformer model is very intensive in its usage of
data requiring at least a million samples to have decent results(which I should be able to meet).
Second, I want to try the GRU (gated recurrent unit) architecture. This architecture is
essentially a predecessor of the transformer model. While it struggles to capture long term
patterns as successfully as the transformer it is significantly easier to train and requires less data.
This could be a way to test to make sure the other parts of my training process are working
correctly.
I will use the ARIA dataset. This dataset contains nearly a million files of solo piano
performance comprising 100,000 hours + of playing time. I will tokenize this data using a python
library called MidiTok. It gives me options to choose how granular to create the data (how many
per beat) and can split them into sequences by setting a token limit. I should end up with well
over a million training examples.
The model is trained in a self supervised manner as the goal is to predict the token in the
sequence that comes next. I will use a form of cross entropy called label smoothing cross entropy
to rate how good the model predicts the next token. I hope to train this on the rcc which gives me
a limit of 3 days to train. I will be sure to save checkpoints so that I can continue training as
necessary. Initially I will train with fewer examples with the GRU architecture on my own
machine. If all goes well I will increase the datasize and use the transformer model on the rcc.

## Dataset Info

I will be using the aria-midi dataset. This dataset is publicly available for download on
HuggingFace and consists of over a million MIDI (musical instrument digital interface) files
obtained by an audio to MIDI encoder model. A MIDI file is essentially a collection of time
stamped events which can be used as input to electronic instruments to control which notes it
plays. There are several subsets of the datasets to choose from and I will choose the pruned
subset which is recommended by the dataset author as the best for generative modelling because
it has been heuristically post filtered to remove MIDI files that contain strange artifacts such a
long periods of silence or repetitive content. Since the Pruned dataset still contains 800,000 +
files I will divide it further into genre subsets. This is a simple process of iterating through the
json and finding entries with a specific genre tag. I will then copy those MIDI files into a genre
subfolder. Initially, I will use the genre jazz to train but optimally with time I will be able to train
models focusing on other genres.
Since MIDI files still are not a good input for the sequential models I will be using to
perform music generative modelling I will make use of a python package known as MidiTok.
This offers tokenizers that will filter out the notes from other potential messages found in MIDI
files and split them into sequences appropriate for training. The tokenizers can be trained in a
single line of code or pulled directly from the hugging face hub. MidiTok is built to work
conveniently with pytorch so after tokenization the dataset can be used with dataloader to create
batches to train off of
