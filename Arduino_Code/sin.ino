//Definición de pesos y bias

//Pesos y bias de la primera capa oculta (1 entrada, 8 neuronas)
float pesos_capa1[8] = {1.1974772, 0.39075312, -0.2973142, 0.36448127, -0.14380677, 0.3485643, 0.35319132, 0.32441604};
float bias_capa1[8]  = {-0.4415212, -0.56996965, 0.00031136768, 0.17679802, 0.23533848, -0.47186759, -0.97312254, -1.7557406};

//Pesos y bias de la segunda capa oculta (8 entradas, 4 neuronas)
float pesos_capa2[8][4] = {
  {0.533144176, -0.0177129135, 0.699099064, 0.683755815},
  {0.0848843083, -0.459298253, -0.871821046, -0.529598713},
  {-0.477367878, 0.181230739, 0.162416369, 0.285663486},
  {0.0159487911, 0.0281045735, 0.000824739982, 0.378508806},
  {0.0753338784, -0.265241265, 0.498325109, -0.470045716},
  {0.0571306162, -0.0180169456, -0.693049431, -0.44951573},
  {-0.76172632, 0.376388907, -0.616057396, -0.112161785},
  {0.529746652, 1.02872527, -1.18166304, 1.17163050}
};
float bias_capa2[4] = {0.17854607, 0.13298722, 0.14456552, -0.02256603};

//Pesos y bias de la capa de salida (4 entradas, 1 neurona)
float pesos_salida[4] = {0.6762751, 0.7105893, 1.4280554, 1.4425309};
float bias_salida = 0.15396428;

//Función de activación tanh
float tanh_activation(float x) {
  return tanh(x);
}

//Función para calcular la salida de la red 
float forward_pass(float entrada) {
  float salida_capa1[8];
  float salida_capa2[4];
  
  //Cálculo de la primera capa oculta 
  for (int i = 0; i < 8; i++) {
    float neta = entrada * pesos_capa1[i] + bias_capa1[i];  
    salida_capa1[i] = tanh_activation(neta);  
  }

  //Cálculo de la segunda capa oculta 
  for (int j = 0; j < 4; j++) {
    float neta = bias_capa2[j]; 
    for (int i = 0; i < 8; i++) {
      neta += salida_capa1[i] * pesos_capa2[i][j];  
    }
    salida_capa2[j] = tanh_activation(neta);  
  }

  //Cálculo de la capa de salida 
  float neta_salida = bias_salida; 
  for (int j = 0; j < 4; j++) {
    neta_salida += salida_capa2[j] * pesos_salida[j];  
  }
  
  return neta_salida;  
}


void setup() {
  Serial.begin(115200);
}


void loop() {
  for (float entrada = 0; entrada <= 2 * 3.1416; entrada += 0.1) {  
    float resultado_red = forward_pass(entrada);  // Salida de la red neuronal
    float resultado_real = sin(entrada);  // Salida de sin(x)

    Serial.print(resultado_red);  // Salida de la red
    Serial.print("\t");  
    Serial.println(resultado_real);  // Salida real de sin(x)

    delay(100); 
  }

}
