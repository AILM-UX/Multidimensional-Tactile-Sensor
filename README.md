<h1 align="center">Multimodal Perception and Three-Dimensional Force Decoupling Method for a Multidimensional Bionic Tactile Sensor</h1>
<div align=center>

 <p align="center">Dapeng Chen, Peng Gao, Zhenjie Ma, Zhou Zhuang, Zhangjia Deng, Hui Zhang, Yun Ling, Xuhui Hu, Hong Zeng, Lina Wei, Jia Liu, Qiang Zhao, Aiguo Song</p>
  <p align="center">Nanjing University of Information Science & Technology</p>
  ---
</div>
  
  Sensor model, fabrication method, control code, and model code. The following two images show the structure and prototype of our sensor.
<div align=center>
<img src="https://github.com/AILM-UX/Multidimensional-Tactile-Sensor/blob/main/figure/fig1a.jpg" alt="Image text" width="600" height="550"/>     <img src="https://github.com/AILM-UX/Multidimensional-Tactile-Sensor/blob/main/figure/fig1b.jpg" alt="Image text" width="400" height="350"/> 
</div>



## The Sensor folder contains the schematic diagrams of the sensors we designed 
This sensor employs a multi-layer design strategy similar to human skin. From top to bottom, it consists of a encapsulation layer, a sensing layer, and a substrate layer, enabling it to simultaneously detect pressure, friction, and temperature.

## Material and Methods

### Fabrication of conductive polymer force-sensing units
CB powder (Tanfeng Tech. Inc, China), MWCNTs powder (NACATE, China), and deionized water were mixed in a mass ratio of 5:2:100 and stirred thoroughly for 15 minutes to form a uniform dispersion. The PU sponge was then fully immersed in the dispersion for 1 hour to ensure that the conductive fillers formed a stable, continuous conductive network within the porous matrix. Subsequently, the impregnated PU sponge was dried in a vacuum oven at 70 °C for 3 hours and cut into 6 mm × 6 mm pieces to serve as independent force-sensing units.

### Fabrication of electrodes and temperature-sensing unit
The electrodes and temperature sensing units are integrated onto a flexible printed circuit board. Four independent and symmetrically distributed planar electrode regions are designed on the FPC, with each group corresponding to a force-sensing unit used to collect signals representing changes in the resistance of the conductive polymer during loading. A digital temperature sensor chip (BMP280, Bosch Semiconductors) is soldered with solder paste and mounted at the center of the FPC. It is connected to the signal acquisition system via a standard digital communication interface to enable synchronous temperature data acquisition.

### Fabrication of the sensor encapsulation layer and Substrate, and sensor assembly
During the preparation of the sensor’s encapsulation layer and substrate, the PDMS (Sylgard 184, Dow Corning) base material and curing agent were thoroughly mixed in a 10:1 ratio, poured into a 3D-printed PLA mold, and then vacuum-cured in a vacuum oven for 2 hours to completely remove air bubbles. The assembly was then cured at 70°C for 3 hours. After demolding, the PDMS encapsulation layer and substrate were obtained. Next, a conductive silver paste (CD02, Conduction) with a volume resistivity of 2×10⁻⁴ Ω·cm was used to bond the CB/MWCNTs/PU conductive polymer to the planar electrodes of the FPC; Finally, silicone rubber (V-705, Valigoo) was used to bond the various layers of the sensor, completing the overall assembly.

## Code
Specifically, “ADS1256” refers to the 24-bit ADC acquisition module code, “BMP280” refers to the temperature acquisition module code, “FT-CFRF” refers to the 3D force decoupling model code, and “Bi-LSTM hardness” refers to the hardness block classification model code.

## Figure
The figure section includes photos of our sensor models and prototypes, evaluation metrics for the three-force decoupling model, and prediction results for some of the test datasets.

