<h1>Traffic Sign Recognition </h1>

<p>This repository contains the code and resources for building a Convolutional Neural Network (CNN) to classify traffic signs using the GTSRB (German Traffic Sign Recognition Benchmark) dataset. The model is implemented using TensorFlow and Keras.</p>

<h2>Table of Contents</h2>
<ul>
  <li><a href="#overview">Overview</a></li>
  <li><a href="#dataset">Dataset</a></li>
  <li><a href="#model-architecture">Model Architecture</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#results">Results</a></li>
  <li><a href="#references">References</a></li>
</ul>

<h2 id="overview">Overview</h2>
<p>The project aims to classify traffic signs into 43 different categories using a deep learning approach with CNN. The trained model can predict the class of a traffic sign image provided as input.</p>

<h2 id="dataset">Dataset</h2>
<p>The dataset used in this project is the <a href="http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset">GTSRB dataset</a>. It contains over 50,000 images of traffic signs belonging to 43 different classes.</p>

<h2 id="model-architecture">Model Architecture</h2>
<p>The CNN model consists of multiple convolutional layers followed by max-pooling and batch normalization. It uses dropout regularization to prevent overfitting and ends with a softmax layer for classification. The model is trained using the Adam optimizer.</p>

<h2 id="installation">Installation</h2>
<pre><code>git clone https://github.com/yourusername/traffic-sign-classification.git
cd traffic-sign-classification
pip install -r requirements.txt
</code></pre>

<h2 id="usage">Usage</h2>
<p>To train the model and evaluate it:</p>
<pre><code>python train.py
</code></pre>

<p>To predict a traffic sign using a saved model:</p>
<pre><code>python predict.py --image_path /path/to/image.png --model_path /path/to/saved_model.h5
</code></pre>

<h2 id="results">Results</h2>
<p>After training for 30 epochs, the model achieved an accuracy of over 95% on the test set. Below are some sample predictions made by the model:</p>

<h2 id="references">References</h2>
<ul>
  <li><a href="http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset">GTSRB Dataset</a></li>
  <li><a href="https://www.tensorflow.org/">TensorFlow</a></li>
  <li><a href="https://keras.io/">Keras Documentation</a></li>
</ul>
