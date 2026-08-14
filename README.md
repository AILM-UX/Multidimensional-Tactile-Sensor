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
During the fabrication of the sensor encapsulation layer and substrate, the PDMS base and curing agent (Sylgard 184, Dow Corning) were thoroughly mixed at a mass ratio of 10:1 and poured into a 3D
printed PLA mold. The mixture was then degassed under vacuum for 2 h to remove trapped air bubbles. Subsequently, the PDMS was cured at 70 ◦C for 3 h and demolded to obtain the encapsulation layer and substrate. Approximately 1 mm-thick thermally conductive silicone was then filled into
the predefined opening in the encapsulation layer to form a localized thermally conductive structure above the temperature-sensing region. Next, the CB/MWCNTs/PU conductive polymer was bonded to the planar electrodes of the FPC using conductive silver paste (CD02, Conduction) with a volume resistivity of 2×10−4 Ω·cm. Finally, the individual sensor layers were bonded using silicone rubber (V-705, Valigoo) to complete the overall assembly.

### Sensor Data Acquisition System
The sensor data acquisition system mainly consists of voltage-divider circuits, an ADS1256 analog-to-digital conversion module, a BMP280 temperature acquisition unit, an STM32F103 microcontroller, and a host computer.  The four piezoresistive force-sensing units are connected to independent voltage-divider circuits to convert force-induced resistance variations into voltage signals, which are subsequently acquired through the 24-bit highprecision ADS1256 analog-to-digital converter. The BMP280 outputs temperature data through a digital communication interface.
The STM32F103 reads the four-channel voltage data acquired by the ADS1256 together with the BMP280 temperature data, integrates the multichannel information into data frames, and transmits them to the host computer through a serial interface. The maximum update rate of a complete data frame is approximately 125 Hz, corresponding to an update period of approximately 8 ms. The host computer performs data reception, real-time visualization, and storage, and runs the three-dimensional force decoupling model to output the three contact force components Fx, Fy, and Fz. The FTTransformer-CFRF model requires approximately 2.01 ms for single-sample inference (Table S2), allowing the prediction of the current frame to be completed before the arrival of the subsequent frame and thereby supporting tactile monitoring and robotic interaction tasks with control periods of approximately 10 ms or longer.
To further suppress occasional spikes and high-frequency random noise, a five-point median filter followed by an exponentially weighted moving average (EWMA) filter was applied to the raw signals on the host computer, with the EWMA smoothing coefficient set to 0.45. The median filter was primarily used to remove transient outliers, whereas the EWMA filter suppressed highfrequency fluctuations during continuous sampling while preserving the stepresponse characteristics associated with loading and unloading.
## Code
Specifically, “ADS1256” refers to the 24-bit ADC acquisition module code, “BMP280” refers to the temperature acquisition module code, “FT-CFRF” refers to the 3D force decoupling model code, and “Bi-LSTM hardness” refers to the hardness block classification model code.

## Figure
The figure section includes photos of our sensor models and prototypes, evaluation metrics for the three-force decoupling model, and prediction results for some of the test datasets.

