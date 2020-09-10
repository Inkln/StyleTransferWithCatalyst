# Style transfer with catalyst

<div align="center">

[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)](https://github.com/catalyst-team/catalyst)

</div>

This repository shows experiment in realtime style transfer with catalyst deep learning framework. The experiment is based on article 
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)


## Examples

<div align="center">
<img src="images/city_origin.jpg" width="400"/>
<img src="images/city_blue.png" width="400"/><br>
<img src="images/city_iris.png" width="400"/>
<img src="images/city_city.png" width="400"/>
</div>

## Training

1. Install dependencies 
```pip3 install torch==1.6.0 catalyst==20.8.2 numpy tensorflow==2.0.0 tensorboard```
    ##### Attention: Catalyst don't have guaranteed backward compatibility, please, use only specified version.

2. Fill config \
Fill all fields marked by "{SPECIFY}" tag in config.yml

3. Run training 
```
catalyst-dl run --config config.yml --verbose
```

4. Inference: \
See ```infer_catalyst.py``` to find details.