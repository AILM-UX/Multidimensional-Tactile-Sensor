#include "delay.h"
#include "sys.h"
#include "usart.h"
#include "ADS1256.h"
#include "BMP280.h"

void delay_ms_soft(uint32_t ms)
{
    uint32_t i;
    for (i = 0; i < ms; i++)
    {
        unsigned char a, b, c;
        for (c = 1; c > 0; c--)
            for (b = 222; b > 0; b--)
                for (a = 12; a > 0; a--);
    }
}

uint8_t g_type = 0; // 0 = single-ended 8-channel, 1 = differential 4-channel

int main(void)
{
    int32_t adc_val[4];
    int32_t volt_val[4];
    double voltage[4];
    double R_sensor[4];
    double temperature;

    const double Vin = 3.3;       
    const double R_fixed = 1000;  
    uint8_t i;
    uint8_t bmp_id;

    delay_init(72);
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
    uart_init(230400);

    /* BMP280 初始化 */
    Bmp_Init();
    bmp_id = BMP280_ReadID();
    printf("\r\nBMP280 ID = 0x%02X\r\n", bmp_id);

    /* ADS1256 初始化 */
    bsp_InitADS1256();
    ADS1256_ReadId();

    printf("\r\nPGA gain = 1, sample rate = 1000SPS, single-ended mode\r\n");
    ADS1256_CfgADC(ADS1256_GAIN_1, ADS1256_1000SPS);

    printf("Start single-ended scan mode, output CH0~CH3 resistance & temperature.\r\n");
    ADS1256_StartScan(0);

    delay_ms_soft(100);

    while (1)
    {
        for (i = 0; i < 4; i++)
        {
           
            adc_val[i] = ADS1256_GetAdc(i);

    
            volt_val[i] = ((int64_t)adc_val[i] * 2500000) / 4194303;

           
            voltage[i] = volt_val[i] / 1000000.0;

      
            if (voltage[i] > 0.0001 && voltage[i] < Vin)
            {
                R_sensor[i] = R_fixed * voltage[i] / (Vin - voltage[i]);
            }
            else
            {
                R_sensor[i] = 0;
            }
        }


        temperature = BMP280_Get_Temperature();

       printf("R0=%.2f R1=%.2f R2=%.2f R3=%.2f T=%.2fC\r\n",
               R_sensor[0],
               R_sensor[1],
               R_sensor[2],
               R_sensor[3],
               temperature);

    }
}