Authors: Markus Laubenthal, Lennard Alms

execute with:

  python tnn.py [h1, [h2, [h3,....]]]

hn = number of neurons in layer n

example:
python tnn.py 5 3

- reads file PA-B-train-01.dat
- Creates following layers:
  - input layer: N (from file)
  - Hidden Layer h1 with 5 neurons
  - Hidden Layer h2 with 3 neurons
  - Output layer: M (from file)


How to edit transfer function:
  See line 96 in code

  Also the derivative of this function has to be edited in line 128/129
  with line 128 being the logistic function and 129 the tanh function
  other derivatives need to be calculated manually since we temporarily
  save the output of the feedforward to accelerate the backprop.
