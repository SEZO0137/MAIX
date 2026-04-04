# MAIX
A Simple Guide To Make Own AI
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
╔══════════════════════════════════════════════════════════════════════════════╗
║           THE COMPLETE BEGINNER'S TOUR TO AI, ML, AND PYTHON               ║
║         Written so simply that a 1-year-old's parent can understand it      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Written for: Someone who has never seen AI code before.
Examples use: Python 3, scikit-learn, numpy, tkinter.
No math degree needed. Everything is explained from scratch.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLE OF CONTENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PART 1 — What Is AI?
  PART 2 — What Is a Model (The Brain)?
  PART 3 — What Are X and Y? (Training Data)
  PART 4 — How Does Training Work? (model.fit(X, y))
  PART 5 — How Does It Learn From Numbers Like 1, 2, 3, 4?
  PART 6 — How Does It Learn From AUDIO?
  PART 7 — How Does It Learn From IMAGES?
  PART 8 — What Are Neurons? (The Fake Brain Cells)
  PART 9 — What Is a Neural Network? (Many Neurons Together)
  PART 10 — What Is the Forward Pass? (How the Brain Thinks)
  PART 11 — What Is Training? (How the Brain Learns)
  PART 12 — What Is ReLU and Softmax? (The Brain's Switches)
  PART 13 — What Is Reinforcement Learning (RL)?
  PART 14 — What Is the Q-Table?
  PART 15 — What Is the RL Formula? (The Reward Equation)
  PART 16 — What Is sklearn? (Your AI Toolkit)
  PART 17 — What Is StandardScaler? (Making Numbers Fair)
  PART 18 — What Is a Pipeline?
  PART 19 — Tkinter — Making Windows and Buttons
  PART 20 — Putting It All Together (Your Word AI Project)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — WHAT IS AI?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AI stands for Artificial Intelligence.

"Artificial" = made by humans, not natural.
"Intelligence" = ability to learn and make decisions.

So AI = a computer program that can LEARN from examples and make decisions.

BABY EXAMPLE:
─────────────
Imagine you show a baby 100 pictures of dogs and say "dog" each time.
Then you show a cat and ask "what is this?"
The baby says "not a dog" or maybe even "cat" if it has seen cats before.

That is EXACTLY what AI does. It looks at examples, finds patterns, and
uses those patterns to answer questions about new things it has never seen.

The difference from normal programs:
  Normal program: YOU write the rules. (if color is brown and has 4 legs → dog)
  AI program:     YOU give examples. The computer FINDS the rules itself.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — WHAT IS A MODEL (THE BRAIN)?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A MODEL is the AI's brain. It is a big collection of NUMBERS (called weights).

Think of it like a recipe book:
  - Before training: the recipe book is empty.
  - After training: the recipe book is full of rules the AI learned.

A model does one thing: you put something IN, it gives you an answer OUT.

  IN  →  [MODEL]  →  OUT

Example:
  IN  = a 1-second recording of someone saying "hello"
  OUT = "that word was: hello" (with 94% confidence)

The model does NOT magically know things. It only knows what you taught it.
If you never showed it the word "hello", it cannot recognise it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — WHAT ARE X AND Y? (TRAINING DATA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

X = the INPUT data (what you give to the model)
y = the ANSWER (what the correct output should be)

Together, X and y are called the TRAINING DATA.

SIMPLE EXAMPLE — Learning fruit:
─────────────────────────────────

  X (what you see)        y (the answer)
  ─────────────────────   ──────────────
  [round, red, sweet]  →  "apple"
  [long, yellow, soft] →  "banana"
  [round, orange, sour]→  "orange"

X is a LIST OF LISTS. Each inner list is one example (one "row" of data).
y is a LIST of answers. Each answer matches one row in X.

IN PYTHON:

  X = [
    [1, 0, 1],   ← round=1, red=0 wait, let's use real numbers
    [0, 1, 0],
    [1, 0, 0],
  ]

  y = [0, 1, 2]  ← 0=apple, 1=banana, 2=orange

The model.fit(X, y) call reads through every row of X, looks at the matching
answer in y, and adjusts its internal numbers (weights) to get better at
predicting the right answer.

HOW BIG IS X?
─────────────
X has SHAPE: (number of examples, number of features)

  X.shape = (100, 26)
  ↑               ↑
  100 recordings  26 numbers describe each recording (MFCC features)

y.shape = (100,)
  ↑
  100 labels (one per recording)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — HOW DOES TRAINING WORK? (model.fit(X, y))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP BY STEP:

Step 1: Start with RANDOM weights (the brain knows nothing)
Step 2: Feed row 0 of X into the model → get a guess
Step 3: Compare guess to y[0] (the right answer)
Step 4: Measure how wrong the guess was → this is called the LOSS or ERROR
Step 5: Nudge the weights a tiny bit to make the next guess better
Step 6: Repeat steps 2-5 for every row in X
Step 7: Do the whole thing many times (each full pass = 1 EPOCH)
Step 8: After many epochs, the weights are good → training is done

BABY ANALOGY:
─────────────
You are teaching a kid to throw a ball into a basket.

  - Kid throws (random guess)
  - Ball misses by 2 metres to the left (error = 2m left)
  - You say "throw more to the right next time" (adjust weights)
  - Kid throws again (better guess)
  - Repeat until the ball goes in the basket consistently

In AI, "how wrong" is measured by a LOSS FUNCTION.
"How to nudge the weights" is done by an algorithm called GRADIENT DESCENT.

The model.fit(X, y) call does ALL of this automatically.
You just give it data. It figures out the weights on its own.

REAL CODE EXAMPLE:

  from sklearn.neural_network import MLPClassifier

  # Create the brain (empty model)
  model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)

  # Teach it (X = examples, y = correct answers)
  model.fit(X, y)

  # Ask it a question
  answer = model.predict([new_example])

That is the whole thing. Three lines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — HOW DOES IT LEARN FROM NUMBERS LIKE 1, 2, 3, 4?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your question: "If I have 4 classes (1, 2, 3, 4), how does model.fit(X, y) work?"

The labels y = [1, 2, 3, 4] are just names. The AI does NOT care that 1 < 2 < 3.
It treats them as CATEGORIES, like "cat", "dog", "bird", "fish".

INTERNALLY the model converts them to a format called ONE-HOT:

  Label 1  →  [1, 0, 0, 0]   ← "it's class 1, definitely not 2, 3, or 4"
  Label 2  →  [0, 1, 0, 0]   ← "it's class 2"
  Label 3  →  [0, 0, 1, 0]   ← "it's class 3"
  Label 4  →  [0, 0, 0, 1]   ← "it's class 4"

The model's output (softmax) also gives 4 numbers that sum to 1.0:

  Model output for some audio → [0.05, 0.87, 0.06, 0.02]
                                     ↑
                                 class 2 wins! (87% confidence)

So when you say:
  y = [1, 1, 1, 2, 2, 2, 3, 3, 3]   (3 recordings of each of 3 classes)

The model fits by:
  1. Taking recording 0 (label=1) → guesses [0.33, 0.33, 0.33] (knows nothing)
  2. Correct answer should be [1, 0, 0]
  3. Error is large → nudge weights
  4. Taking recording 3 (label=2) → guesses [0.31, 0.38, 0.31] (slightly better)
  5. ...and so on for 500 epochs

After 500 passes through all the data, it gets good at telling apart class 1, 2, 3.

COMPLETE CODE — 4 classes of fruit hardness:

  import numpy as np
  from sklearn.neural_network import MLPClassifier

  # X = features (e.g. [hardness, size, sweetness])
  X = np.array([
    [9, 5, 8],   # example of class 1 (apple)
    [8, 5, 7],   # example of class 1 (apple)
    [2, 8, 9],   # example of class 2 (banana)
    [3, 7, 8],   # example of class 2 (banana)
    [7, 4, 3],   # example of class 3 (lemon)
    [8, 4, 2],   # example of class 3 (lemon)
    [5, 9, 6],   # example of class 4 (melon)
    [4, 9, 7],   # example of class 4 (melon)
  ])

  # y = labels (just numbers, 1 per row in X)
  y = np.array([1, 1, 2, 2, 3, 3, 4, 4])

  # Build and train model
  model = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=500)
  model.fit(X, y)

  # Predict a new fruit
  new_fruit = [[7, 5, 8]]          # hardness=7, size=5, sweetness=8
  prediction = model.predict(new_fruit)
  print(prediction)                # → [1]  (model guesses: apple!)

  # Get probabilities for all 4 classes
  probs = model.predict_proba(new_fruit)
  print(probs)                     # → [[0.82, 0.05, 0.08, 0.05]]
  #                                          ↑
  #                                     82% confident it's class 1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — HOW DOES IT LEARN FROM AUDIO?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AI models only understand NUMBERS. Audio is sound waves. So we must convert
the audio into numbers first. This conversion is called FEATURE EXTRACTION.

WHAT IS A SOUND WAVE?
─────────────────────
Sound is vibrations in the air. A microphone turns vibrations into numbers
(thousands of numbers per second). At 16,000 Hz sample rate, 1 second of
audio = 16,000 numbers.

  [0.01, -0.03, 0.05, -0.02, 0.08, ...]  ← raw audio (16000 numbers)

But 16,000 numbers is too many and too noisy. We need to summarise the sound.

STEP 1 — SPLIT INTO FRAMES:
────────────────────────────
We chop the audio into small overlapping chunks called FRAMES.
  Frame size = 256 samples (= 16ms of sound)
  Hop size   = 128 samples (= 8ms between frames)
  1 second → about 124 frames

STEP 2 — FFT (FAST FOURIER TRANSFORM):
───────────────────────────────────────
Apply FFT to each frame. FFT answers: "which frequencies are in this sound?"

  A pure "A" note  → lots of energy at 440 Hz, nothing elsewhere
  Human speech "a" → energy spread across many frequencies (complex pattern)

FFT turns 256 time-domain numbers into 129 frequency numbers.

STEP 3 — MEL FILTERBANK:
────────────────────────
Human ears don't hear all frequencies equally. We hear low frequencies
better than high frequencies. The Mel scale mimics human hearing.

We group the 129 FFT bins into 26 Mel bands. This gives 26 numbers per frame.

STEP 4 — LOGARITHM:
────────────────────
We take log() of the Mel energies. This matches how humans perceive loudness
(we hear differences in quiet sounds better than differences in loud sounds).

STEP 5 — DCT (MFCC):
─────────────────────
We apply a DCT (similar to compression) to get 13 MFCC coefficients per frame.
These 13 numbers compactly describe the "shape" of the sound in that frame.

STEP 6 — MEAN AND STD:
────────────────────────
Across all 124 frames, we take:
  mean of each MFCC coefficient → 13 numbers
  std  of each MFCC coefficient → 13 numbers
  Total = 26 numbers per recording

These 26 numbers are the FEATURE VECTOR — the audio boiled down to 26 numbers
that the model can learn from.

COMPLETE AUDIO PIPELINE:

  1 second audio (16000 numbers)
        ↓  split into frames
  124 frames × 256 samples
        ↓  FFT each frame
  124 frames × 129 frequency bins
        ↓  Mel filterbank
  124 frames × 26 Mel energies
        ↓  log
  124 frames × 26 log-Mel values
        ↓  DCT
  124 frames × 13 MFCCs
        ↓  mean + std across all frames
  26 numbers  ← final feature vector for this 1 second clip

Then:  model.fit(X, y)  where X has shape (n_recordings, 26)

CODE — extract 26 features from audio:

  import numpy as np

  def extract_features(audio):
      # audio = 1D array of 16000 float32 values

      N_FFT = 256
      HOP   = 128
      mfcc_frames = []

      for i in range(0, len(audio) - N_FFT, HOP):
          frame = audio[i : i + N_FFT]
          # FFT → power spectrum
          spectrum = np.fft.rfft(frame)
          power    = np.abs(spectrum) ** 2
          # (In real code: also apply Mel filterbank and DCT here)
          mfcc_frames.append(power[:13])   # simplified: just use first 13 bins

      mfcc_frames = np.array(mfcc_frames)  # shape: (n_frames, 13)
      mean = mfcc_frames.mean(axis=0)      # shape: (13,)
      std  = mfcc_frames.std(axis=0)       # shape: (13,)

      return np.concatenate([mean, std])   # shape: (26,)

  # Use it:
  raw_audio = np.random.randn(16000).astype(np.float32) * 0.1
  features  = extract_features(raw_audio)
  print(features.shape)   # → (26,)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — HOW DOES IT LEARN FROM IMAGES?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

An image is just a grid of pixels. Each pixel is 1-3 numbers.

  Greyscale image (100×100 px): 10,000 numbers
  Colour image   (100×100 px): 30,000 numbers (R, G, B per pixel)

SIMPLE WAY — FLATTEN:
──────────────────────
Just stretch the 2D grid into a 1D list of numbers.

  100×100 image → [255, 230, 200, 255, 128, ...]   (10,000 numbers)

Then treat those 10,000 numbers as features, same as audio.

  model.fit(X_images, y_labels)

  where X_images.shape = (n_images, 10000)

BETTER WAY — CNN (Convolutional Neural Network):
─────────────────────────────────────────────────
Instead of flattening, a CNN slides a small "filter" (3×3 window) across
the image to detect edges, shapes, and patterns. This is much smarter
because it understands that nearby pixels belong together.

But the basic idea is the same: image → numbers → model → prediction.

CODE — image to flat features:

  from PIL import Image
  import numpy as np

  img   = Image.open("cat.jpg").convert("L")   # convert to greyscale
  img   = img.resize((28, 28))                 # make it 28×28 pixels
  arr   = np.array(img)                        # shape: (28, 28)
  flat  = arr.flatten() / 255.0               # shape: (784,)  values 0-1
  # Now 'flat' is your feature vector for this image

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — WHAT ARE NEURONS? (THE FAKE BRAIN CELLS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A real brain has neurons connected by synapses.
A fake (artificial) neuron is much simpler. It is just MULTIPLICATION and ADDITION.

One neuron does this:

  output = activation( (input1 × weight1) + (input2 × weight2) + bias )

That's it. Let's make it concrete:

  Inputs:   [0.5, 0.3, 0.8]          ← three numbers coming in
  Weights:  [0.2, 0.9, 0.4]          ← three numbers the neuron learned
  Bias:     0.1                       ← one extra number the neuron learned

  Step 1:  (0.5×0.2) + (0.3×0.9) + (0.8×0.4)  =  0.1 + 0.27 + 0.32  =  0.69
  Step 2:  0.69 + 0.1 (add bias)               =  0.79
  Step 3:  activation(0.79)                     =  0.79  (if using ReLU)

  output = 0.79

The weights and bias are the numbers that TRAINING changes.
Before training: random weights.
After training: weights that make the output correct.

PYTHON VERSION:

  import numpy as np

  inputs  = np.array([0.5, 0.3, 0.8])
  weights = np.array([0.2, 0.9, 0.4])
  bias    = 0.1

  output = np.dot(inputs, weights) + bias
  # np.dot multiplies each pair and adds them up (same as step 1+2)
  # output = 0.79

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — WHAT IS A NEURAL NETWORK? (MANY NEURONS TOGETHER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A neural network is many neurons arranged in LAYERS.

  INPUT LAYER → HIDDEN LAYER 1 → HIDDEN LAYER 2 → OUTPUT LAYER

  [26 inputs] → [64 neurons] → [32 neurons] → [4 outputs]

Each layer's neurons look at ALL the outputs of the previous layer.
This means 26 inputs × 64 neurons = 1,664 connections in just layer 1.
Each connection has its own weight. That's 1,664 numbers to learn.

The WEIGHT MATRIX of a layer is a 2D array:
  Layer 0 weight matrix shape:  (26, 64)   ← 26 inputs, 64 outputs
  Layer 1 weight matrix shape:  (64, 32)   ← 64 inputs, 32 outputs
  Layer 2 weight matrix shape:  (32, 4)    ← 32 inputs, 4 class outputs

CALCULATING ONE LAYER IN CODE:

  import numpy as np

  x       = np.array([...])          # shape: (26,)   — input
  W0      = np.random.randn(26, 64)  # shape: (26, 64) — weight matrix
  b0      = np.zeros(64)             # shape: (64,)   — bias vector

  h1 = x @ W0 + b0                  # shape: (64,)
  #    ↑
  # @ means matrix multiply
  # Each of the 64 output values is a weighted sum of all 26 inputs

  h1 = np.maximum(h1, 0)            # ReLU activation (see Part 12)

This is EXACTLY what the model does inside when you call model.predict().
The weights W0, b0, W1, b1, W2, b2 are what model.fit() learns.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 10 — WHAT IS THE FORWARD PASS? (HOW THE BRAIN THINKS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When you call model.predict(x), internally it runs a FORWARD PASS:

  x       →  Layer 0  →  h1  →  Layer 1  →  h2  →  Layer 2  →  output
  (26 #s)    (26→64)    (64)     (64→32)    (32)     (32→4)     (4 #s)

CODE — manual forward pass (same as what the firmware does):

  import numpy as np

  def relu(x):
      return np.maximum(x, 0)   # negative → 0, positive → keep

  def softmax(x):
      e = np.exp(x - x.max())   # subtract max for numerical safety
      return e / e.sum()         # make all values sum to 1.0

  def predict(input_features, W0, b0, W1, b1, W2, b2):
      h1  = relu(input_features @ W0 + b0)    # hidden layer 1
      h2  = relu(h1             @ W1 + b1)    # hidden layer 2
      out = softmax(h2          @ W2 + b2)    # output probabilities
      return out

  # Call it:
  probs = predict(my_features, W0, b0, W1, b1, W2, b2)
  # probs = [0.05, 0.87, 0.06, 0.02]
  # → model is 87% confident it's class 2 (index 1)

  prediction = probs.argmax()   # → 1  (the index of the highest value)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 11 — WHAT IS TRAINING? (HOW THE BRAIN LEARNS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training = finding the best values for W0, b0, W1, b1, W2, b2.

It uses an algorithm called GRADIENT DESCENT. Here is the idea:

ANALOGY — FINDING THE LOWEST VALLEY BLINDFOLDED:
─────────────────────────────────────────────────
Imagine you are on a hilly landscape, blindfolded.
Your goal is to reach the lowest valley (= minimum loss = best model).

You can feel the ground under your feet:
  - If the ground slopes down to the left → step left
  - If the ground slopes down to the right → step right
  - Repeat tiny steps until you are at the bottom

The GRADIENT is the slope of the ground at your current position.
GRADIENT DESCENT = always step in the direction that goes downhill.

In AI:
  "ground = loss function"         (how wrong is the model right now?)
  "slope = gradient"               (which direction improves the model?)
  "step size = learning rate"      (how big a step to take)

LEARNING RATE:
──────────────
  Too large: you jump over the valley and miss it
  Too small: takes forever to get there
  Good value: 0.001 is typical (each weight changes by 0.001 × gradient)

THE LOSS FUNCTION:
──────────────────
It measures how wrong the model is. Common one: Cross-Entropy Loss.

  If model predicts [0.9, 0.05, 0.05] and correct answer is class 0 → small loss
  If model predicts [0.1, 0.8,  0.1]  and correct answer is class 0 → big loss

THE UPDATE RULE:
─────────────────
  new_weight = old_weight - learning_rate × gradient

  Example:
    old_weight = 0.5
    gradient   = 2.0      (loss decreases if weight decreases)
    lr         = 0.001

    new_weight = 0.5 - 0.001 × 2.0 = 0.498

So the weight moved a tiny bit (0.5 → 0.498) in the better direction.
After thousands of such updates, weights are good.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 12 — WHAT IS ReLU AND SOFTMAX? (THE BRAIN'S SWITCHES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RELU (Rectified Linear Unit):
──────────────────────────────
  relu(x) = x   if x > 0
  relu(x) = 0   if x ≤ 0

Basically: "if negative, make it zero. If positive, keep it."

WHY? Without activation functions, stacking many layers does nothing useful
(it all collapses to one linear equation). ReLU adds non-linearity,
which lets the network learn complex patterns like "this frequency combined
with that rhythm = the word hello".

CODE:
  import numpy as np
  x    = np.array([-3, -1, 0, 2, 5])
  relu = np.maximum(x, 0)
  # → [0, 0, 0, 2, 5]

SOFTMAX:
─────────
Turns any list of numbers into PROBABILITIES that sum to 1.

  logits = [2.0, 1.0, 0.1]     ← raw outputs of final layer
  softmax → [0.665, 0.245, 0.090]
  # sum = 0.665 + 0.245 + 0.090 = 1.0  ✓

Used in the output layer so you get "how confident" instead of raw numbers.

CODE:
  import numpy as np
  def softmax(x):
      e = np.exp(x - x.max())   # subtract max to avoid overflow
      return e / e.sum()
  probs = softmax(np.array([2.0, 1.0, 0.1]))
  # → [0.665, 0.245, 0.090]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 13 — WHAT IS REINFORCEMENT LEARNING (RL)?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All the AI above was SUPERVISED learning: you give examples with correct answers.
RL is different: the AI learns by TRIAL AND ERROR using REWARDS.

ANALOGY — TRAINING A DOG:
──────────────────────────
  Dog sits when you say "sit" → give treat (reward +1)
  Dog doesn't sit             → no treat   (reward 0)
  Dog bites the sofa          → spray water (reward -1)

Over time the dog learns: "sitting when asked = good. Biting sofa = bad."
Nobody told the dog the rules. It figured it out from rewards.

RL uses exactly this:
  AGENT   = the AI (the "dog")
  STATE   = the current situation (what the AI observes right now)
  ACTION  = what the AI does (e.g., which word to say next)
  REWARD  = feedback (good answer = +1, bad answer = 0)

THE RL LOOP:

  1. Agent observes STATE (e.g., user said "hello")
  2. Agent picks ACTION (e.g., respond with "you")
  3. Environment gives REWARD (+1 if good, 0 if bad)
  4. Agent updates its knowledge
  5. Go to step 1

IN YOUR WORD AI:
  State  = last word the AI heard (e.g., "hello")
  Action = next word to say (e.g., "there" or "good" or "bye")
  Reward = user taps button (single tap = +1, double tap = 0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 14 — WHAT IS THE Q-TABLE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The Q-Table is the agent's memory. It is a 2D table of numbers.

  Rows    = every possible STATE  (e.g., 31 possible "last words heard")
  Columns = every possible ACTION (e.g., 31 possible "words to say next")
  Values  = Q-values (how good is this action from this state?)

EXAMPLE Q-TABLE (tiny version with 3 states, 3 actions):

               Say "hello"   Say "bye"   Say "good"
  heard "hi"  [  0.8      ,   0.2     ,   0.6     ]
  heard "bye" [  0.3      ,   0.9     ,   0.1     ]
  heard "ok"  [  0.5      ,   0.4     ,   0.7     ]

If the user said "hi" (state = "hi"), the AI looks at row "heard hi":
  [0.8, 0.2, 0.6] → "Say hello" has the highest Q-value (0.8)
  → AI says "hello"

After the user taps the button (reward), the AI updates Q["hi"]["hello"].

The AI starts with all Q-values = 0 (or random small numbers).
Over many conversations and taps, good Q-values grow, bad ones stay small.

CODE — simple Q-table:

  import numpy as np

  n_states  = 4   # 4 possible states
  n_actions = 4   # 4 possible actions

  Q = np.zeros((n_states, n_actions))   # all zeros at start

  # After receiving reward, update Q:
  state   = 1    # current state
  action  = 2    # action we took
  reward  = 1.0  # we got a reward!
  Q[state][action] += 0.1 * reward   # simple update

  print(Q)
  # Row 1, column 2 is now 0.1. Everything else is still 0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 15 — WHAT IS THE RL FORMULA? (THE REWARD EQUATION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The core RL update formula is called the BELLMAN EQUATION (Q-learning version):

  Q(s, a)  ←  Q(s, a) + α × [r + γ × max Q(s', a')  -  Q(s, a)]

Let's decode every symbol:

  Q(s, a)          = current Q-value for state s, action a (what we think now)
  ←                = "update to"
  α (alpha)        = learning rate (e.g. 0.15) — how fast to update
  r                = reward we just received (e.g. 1.0 for good answer)
  γ (gamma)        = discount factor (e.g. 0.9) — how much future matters
  s'               = next state (state after taking action a)
  max Q(s', a')    = best Q-value available from the next state
  [...]            = the TD error (temporal difference error)

BABY ANALOGY:
─────────────
You are playing a game. You make a move. You get a cookie (reward=1).
The game is not over — there are more moves to make.

  New estimate  = cookie I got NOW  +  0.9 × best cookie I can get LATER
  Update amount = new estimate - what I previously thought this move was worth
  New Q-value   = old Q-value + 0.15 × update amount

The 0.9 (gamma) means: "future rewards matter, but a cookie now is worth
more than a cookie 3 moves from now." Lower gamma = more short-sighted.

CODE — Q-learning update:

  alpha = 0.15    # learning rate
  gamma = 0.90    # discount factor

  state      = 2           # current state (e.g., heard "hello")
  action     = 5           # action taken  (e.g., said "you")
  reward     = 1.0         # user tapped (good!)
  next_state = 5           # now we are in state 5 (heard "you"?)

  # Current Q-value
  current_q = Q[state][action]

  # Best possible Q-value from the next state
  best_next_q = Q[next_state].max()

  # TD target (what the Q-value should be)
  td_target = reward + gamma * best_next_q

  # TD error (how wrong were we?)
  td_error = td_target - current_q

  # Update the Q-value
  Q[state][action] = current_q + alpha * td_error

  print(f"Q updated from {current_q:.3f} to {Q[state][action]:.3f}")

EXAMPLE:
  current_q    = 0.2   (we thought this action was mediocre)
  reward       = 1.0   (but it was good!)
  best_next_q  = 0.5   (future looks promising too)
  td_target    = 1.0 + 0.9 × 0.5 = 1.45
  td_error     = 1.45 - 0.2 = 1.25
  new Q        = 0.2 + 0.15 × 1.25 = 0.2 + 0.1875 = 0.3875

The Q-value went up from 0.2 to 0.39 because we got rewarded.
Next time the AI is in the same state, it will prefer this action more.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 16 — WHAT IS SKLEARN? (YOUR AI TOOLKIT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sklearn (scikit-learn) is a Python library with ready-made AI tools.
Instead of coding a neural network from scratch, you say:

  from sklearn.neural_network import MLPClassifier

  model = MLPClassifier(hidden_layer_sizes=(64, 32))
  model.fit(X, y)

And it handles all the gradient descent, weight updates, etc. for you.

KEY SKLEARN TOOLS USED IN THIS PROJECT:

  MLPClassifier          — multi-layer neural network for classification
  StandardScaler         — normalises features to mean=0, std=1
  Pipeline               — chains steps together
  cross_val_score        — tests model on unseen data (k-fold)
  StratifiedKFold        — k-fold split that keeps class balance
  classification_report  — shows per-class accuracy, precision, recall

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 17 — WHAT IS STANDARDSCALER? (MAKING NUMBERS FAIR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBLEM: features can have very different ranges.
  Feature 1 (MFCC mean): values like -30 to 5
  Feature 2 (MFCC std):  values like 0 to 2

If feature 1 is 1000× bigger than feature 2, the model pays too much attention
to feature 1 and ignores feature 2. This makes learning worse.

SOLUTION: StandardScaler makes every feature have mean=0 and std=1.

  scaled = (original - mean) / std

EXAMPLE:
  original values: [10, 20, 30, 40, 50]
  mean = 30, std = 14.14

  scaled: [-1.41, -0.71, 0.0, 0.71, 1.41]

Now all features are on the same scale. Learning is much easier.

CODE:

  from sklearn.preprocessing import StandardScaler
  import numpy as np

  X = np.array([[10, 2000],    ← feature 1 is small, feature 2 is huge
                [20, 3000],
                [30, 4000]])

  scaler = StandardScaler()
  scaler.fit(X)          # learn the mean and std of each feature
  X_scaled = scaler.transform(X)   # apply scaling

  print(X_scaled)
  # Both features now have similar range

IMPORTANT: Fit the scaler on TRAINING data only. Then use the SAME scaler
to transform your test/production data. Never fit on test data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 18 — WHAT IS A PIPELINE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A Pipeline chains multiple steps so they happen in order automatically.

WITHOUT pipeline (error-prone):
  scaler.fit(X_train)
  X_train_scaled = scaler.transform(X_train)
  model.fit(X_train_scaled, y_train)
  X_test_scaled = scaler.transform(X_test)   ← easy to forget this!
  model.predict(X_test_scaled)

WITH pipeline (clean and safe):
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.neural_network import MLPClassifier

  pipe = Pipeline([
      ("sc",  StandardScaler()),        ← step 1: scale
      ("clf", MLPClassifier(...)),       ← step 2: train
  ])

  pipe.fit(X_train, y_train)            # scales THEN trains automatically
  pipe.predict(X_test)                  # scales THEN predicts automatically

The pipeline remembers all its steps. When you save it with joblib and
reload it later, all steps (scaler + model) come back together.

  import joblib
  joblib.dump(pipe, "model.pkl")        # save
  pipe2 = joblib.load("model.pkl")     # load — scaler included!
  pipe2.predict(new_data)              # works correctly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 19 — TKINTER — MAKING WINDOWS AND BUTTONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tkinter is Python's built-in GUI (Graphical User Interface) library.
It lets you make windows with buttons, text boxes, labels, etc.
No installation needed — it comes with Python.

── HELLO WORLD WINDOW ──────────────────────────────────────────────────────

  import tkinter as tk           # import the library, call it "tk"

  root = tk.Tk()                 # create the main window
  root.title("My First Window")  # set the window title

  root.mainloop()                # keep the window open (event loop)

Run this. A blank window appears. Close it and the program ends.

── ADDING A LABEL ───────────────────────────────────────────────────────────

  import tkinter as tk

  root = tk.Tk()
  root.title("Labels")

  label = tk.Label(root, text="Hello World!")   # create a text label
  label.pack()                                  # add it to the window

  root.mainloop()

  LINE BY LINE:
  tk.Label(root, ...)  → create a label INSIDE the root window
  text="Hello World!"  → the text to show
  .pack()              → place the label in the window (auto-layout)

── ADDING A BUTTON ──────────────────────────────────────────────────────────

  import tkinter as tk

  def on_click():                               # function to call when clicked
      print("Button was clicked!")

  root = tk.Tk()

  btn = tk.Button(root, text="Click Me", command=on_click)
  btn.pack(pady=10)                             # pady=10 adds space above+below

  root.mainloop()

  LINE BY LINE:
  def on_click():          → define what happens when button is pressed
  tk.Button(root, ...)     → create a button inside root
  text="Click Me"          → what text to show on the button
  command=on_click         → which function to call when clicked
  .pack(pady=10)           → place it with 10px padding top and bottom

── CHANGING LABEL TEXT WHEN BUTTON IS CLICKED ───────────────────────────────

  import tkinter as tk

  root = tk.Tk()

  lbl = tk.Label(root, text="Nothing happened yet", fg="gray")
  lbl.pack(pady=5)

  def on_click():
      lbl.config(text="Button was clicked!", fg="green")
      # .config() changes a widget's properties after it was created

  btn = tk.Button(root, text="Click Me", command=on_click)
  btn.pack(pady=5)

  root.mainloop()

── ENTRY (TEXT INPUT BOX) ───────────────────────────────────────────────────

  import tkinter as tk

  root = tk.Tk()

  tk.Label(root, text="Your name:").pack()

  entry = tk.Entry(root, width=20)   # text box, 20 characters wide
  entry.pack(pady=4)

  def greet():
      name = entry.get()             # .get() reads what the user typed
      tk.Label(root, text=f"Hello, {name}!").pack()

  tk.Button(root, text="Greet", command=greet).pack()

  root.mainloop()

── ARRANGING THINGS IN A GRID ──────────────────────────────────────────────

  import tkinter as tk

  root = tk.Tk()

  # .grid(row=, column=) places widgets in rows and columns
  tk.Label(root, text="Name:").grid(row=0, column=0, sticky="e")
  tk.Entry(root).grid(row=0, column=1)

  tk.Label(root, text="Age:").grid(row=1, column=0, sticky="e")
  tk.Entry(root).grid(row=1, column=1)

  tk.Button(root, text="Submit").grid(row=2, column=0, columnspan=2)

  root.mainloop()

  sticky="e"          → stick to the east (right) side of the cell
  columnspan=2        → span across 2 columns

── COLORS AND FONTS ─────────────────────────────────────────────────────────

  import tkinter as tk

  root = tk.Tk()

  tk.Label(
      root,
      text="Big Red Title",
      font=("Helvetica", 18, "bold"),   # font family, size, style
      fg="red",                          # foreground (text) color
      bg="yellow",                       # background color
  ).pack(padx=20, pady=10)

  tk.Button(
      root,
      text="Green Button",
      bg="#27ae60",     # hex color code (green)
      fg="white",       # white text
      relief="flat",    # flat = no raised border
      padx=10,          # internal left/right padding
  ).pack(pady=5)

  root.mainloop()

── READING KEY PRESSES ──────────────────────────────────────────────────────

  import tkinter as tk

  root = tk.Tk()

  lbl = tk.Label(root, text="Press R", font=("Helvetica", 14))
  lbl.pack(pady=20)

  def on_r_press(event):                  # event contains info about the key
      lbl.config(text="R is DOWN!", fg="red")

  def on_r_release(event):
      lbl.config(text="R released", fg="gray")

  root.bind("<KeyPress-r>",   on_r_press)    # triggers when R is pressed
  root.bind("<KeyRelease-r>", on_r_release)  # triggers when R is released

  root.mainloop()

── SHOWING A POPUP MESSAGE ──────────────────────────────────────────────────

  import tkinter as tk
  from tkinter import messagebox

  root = tk.Tk()

  def show_info():
      messagebox.showinfo("Title", "This is info text!")

  def show_error():
      messagebox.showerror("Error", "Something went wrong.")

  def ask_yes_no():
      answer = messagebox.askyesno("Question", "Do you like AI?")
      print(answer)   # True if Yes, False if No

  tk.Button(root, text="Info",    command=show_info).pack(pady=5)
  tk.Button(root, text="Error",   command=show_error).pack(pady=5)
  tk.Button(root, text="Ask",     command=ask_yes_no).pack(pady=5)

  root.mainloop()

── A COMPLETE MINI APP ──────────────────────────────────────────────────────
  This mini app adds two numbers when you press the button.

  import tkinter as tk
  from tkinter import messagebox

  root = tk.Tk()
  root.title("Number Adder")

  # Row 0: first number
  tk.Label(root, text="Number 1:").grid(row=0, column=0, padx=5, pady=5)
  ent_a = tk.Entry(root, width=10)
  ent_a.grid(row=0, column=1)

  # Row 1: second number
  tk.Label(root, text="Number 2:").grid(row=1, column=0, padx=5, pady=5)
  ent_b = tk.Entry(root, width=10)
  ent_b.grid(row=1, column=1)

  # Row 2: result label
  lbl_result = tk.Label(root, text="Result: —", fg="blue", font=("Helvetica", 12))
  lbl_result.grid(row=2, column=0, columnspan=2, pady=5)

  # Button handler
  def add_numbers():
      try:
          a = float(ent_a.get())    # read first entry as a number
          b = float(ent_b.get())    # read second entry as a number
          lbl_result.config(text=f"Result: {a + b}")
      except ValueError:
          messagebox.showerror("Error", "Please enter valid numbers!")

  # Row 3: Add button
  tk.Button(root, text="Add", command=add_numbers,
            bg="#2980b9", fg="white", relief="flat").grid(
                row=3, column=0, columnspan=2, pady=10)

  root.mainloop()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 20 — PUTTING IT ALL TOGETHER (YOUR WORD AI PROJECT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Here is how every concept above connects in YOUR project:

LISTENING AI (recorder_ui.py + audio_features.cpp):
  1. You speak into the microphone (INMP441 on RP2040, or PC mic)
  2. Audio is captured as 16,000 numbers (1 second at 16kHz)
  3. extract_features() converts it to 26 MFCC numbers (Part 6)
  4. These 26 numbers go into X
  5. Your label (which word you said) goes into y
  6. model.fit(X, y) trains the neural network (Parts 4, 9, 10, 11)
  7. model.pkl is saved with the trained weights (Part 18)

THINKING AI (reinforcement.cpp):
  1. After hearing a word, the RL agent looks up the current STATE in the Q-table
  2. It picks the best ACTION (which word to say) — epsilon-greedy (usually best,
     sometimes random to explore)
  3. It forms a response sentence
  4. User taps button to give REWARD
  5. Q-table updates with Bellman equation (Part 15)
  6. Neural matrix also gets a gradient step (Part 11)
  7. This happens every second too (replay of last experience)

DEPLOYMENT (Arduino / RP2040):
  1. recorder_ui.py writes all weights into model_data.h as C float arrays
  2. You flash the RP2040 with Arduino IDE
  3. The RP2040 runs the forward pass (Part 10) in pure C using those arrays
  4. No Python, no sklearn — just the math, which is just multiply and add

THE FULL CONVERSATION LOOP:

  User says "hello"
      ↓ microphone → MFCC features (26 numbers)
      ↓ inference_run() → 94% class 1 = "hello"
      ↓ sentence buffer: ["hello"]
      ↓ RL agent: state="hello", best action="you" (Q=0.8)
      ↓ OLED shows: "you"
      ↓ User taps button (good!)
      ↓ Q["hello"]["you"] += 0.15 × (1.0 + 0.9×0.5 − 0.8) → goes up
      ↓ Next time someone says "hello", agent will say "you" with more confidence

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK REFERENCE — KEY FORMULAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONE NEURON:
  output = relu( sum(inputs × weights) + bias )

FORWARD PASS (one layer):
  h = relu( X @ W + b )           ← X=input, W=weights, b=bias

FULL NETWORK:
  h1  = relu( x    @ W0 + b0 )
  h2  = relu( h1   @ W1 + b1 )
  out = softmax( h2 @ W2 + b2 )

WEIGHT UPDATE (gradient descent):
  W = W − lr × gradient

Q-TABLE UPDATE (Bellman):
  Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') − Q(s,a)]

MFCC FEATURES:
  audio → FFT → Mel filterbank → log → DCT → mean+std → 26 numbers

STANDARDSCALER:
  scaled = (value − mean) / std

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK REFERENCE — TKINTER CHEAT SHEET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  root = tk.Tk()                          create main window
  root.title("Name")                      set window title
  root.geometry("400x300")               set size in pixels
  root.mainloop()                         keep window open

  tk.Label(parent, text="hi")            text display
  tk.Button(parent, text="ok", command=fn)  clickable button
  tk.Entry(parent, width=20)             text input box
  tk.Frame(parent)                        invisible container
  tk.LabelFrame(parent, text="Section")  bordered section

  widget.pack()                           auto-layout
  widget.pack(fill="x", pady=5)          stretch horizontal, add 5px gap
  widget.grid(row=0, column=1)           grid layout
  widget.grid(sticky="ew")              stretch east-west

  widget.config(text="new text")         change property
  entry.get()                            read text from entry
  entry.delete(0, tk.END)               clear entry

  root.bind("<KeyPress-r>", fn)          detect key press
  messagebox.showinfo("T", "msg")       popup info
  messagebox.showerror("T", "msg")      popup error
  messagebox.askyesno("T", "q")         popup yes/no question

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
END OF GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If you understand everything in this guide, you understand the full Word AI
system from the microphone all the way to the RP2040 display.

The most important insight: EVERYTHING in AI is just numbers.
Audio = numbers. Images = numbers. Weights = numbers. Rewards = numbers.
The "intelligence" is just the pattern of how those numbers relate to each other.

You now know how to:
  ✓ Explain what AI is
  ✓ Understand X and y (training data)
  ✓ Understand how model.fit(X, y) learns
  ✓ Understand how audio becomes numbers (MFCC)
  ✓ Understand how images become numbers (pixels)
  ✓ Understand what a neuron is
  ✓ Understand what a neural network is
  ✓ Understand ReLU and softmax
  ✓ Understand reinforcement learning
  ✓ Understand the Q-table
  ✓ Understand the Bellman equation
  ✓ Build basic Tkinter windows with buttons, labels, and inputs
