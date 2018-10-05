# attribute_grounding
Gender/Age attribute grounding using unsupervised manner.

For a clarification of the file system:

#### Train_attr_attention_embedding.py:
<pre>Training script for attribute grounding.</pre>

#### /Models/Model7.py:
<pre>Attention model for attribute grounding, it's based on a pre-trained Res-50 Network on person gender/age classification network.</pre>

#### /lib/:
<pre>Contains all the neccesary dependencies for our framework, it consists of: 
<ul>
  <li>bilinear pooling module: Implemented from <a href="https://github.com/DeepInsight-PCALab/CompactBilinearPooling-Pytorch">Compact Bilinear Pooling</a>. Faster Fourier Transform module is needed before using. Download and install it from <a href="https://github.com/DeepInsight-PCALab/CompactBilinearPooling-Pytorch">here</a> by running:
 <pre>pip3 install pytorch_fft</pre>
</li>
  <li>resnet: We modified the last fully connected layer from 2048d to 256d to a more compact representation.</li>
  <li>nms/roi_align module: Not neccesary in this time. (For entity grounding and bbox detection.)</li>
</ul></pre>

#### /checkpoint/:
<pre>To store the pre-trained res50 Network, and the overall network.</pre>
To download the pre-trained unsupervised network:
<ul>
  <li><a href="https://drive.google.com/open?id=10syFqPtkUp4frDV6YEQbgbKs9dUdfTB_">Res50</a> can be found it here.</li>
  <li><a href="https://drive.google.com/open?id=1YPkw0n-beGZ1HTCxxroQTMa21nvg613p"> Model 7 in here</a>.</li>
</ul>

#### parser.py:
In order to re_train our framework, several things might be modified:

In parser.py, img_path/annotations need to be changed to your local coco_2017_train directory:
<pre>/path/to/your/local/coco17/image path/annotations/</pre>

Argument resume is for loading pre-trained overall model.
