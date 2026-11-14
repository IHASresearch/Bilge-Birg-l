#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>

// Pin Tanımlamaları
#define CE_PIN 7
#define CSN_PIN 8

RF24 radio(CE_PIN, CSN_PIN);
const byte address[6] = "00001";

// Alıcı ile BİREBİR AYNISI olması gereken veri paketi
struct VeriPaketi {
  byte cihazID;        // Gönderen cihazın numarası (örn: 1)
  float deger;         // Örnek bir sayısal değer / sayaç
  char mesaj[20];      // Gönderilecek metin (Maksimum 32 byte limitine dikkat edilmeli)
};

VeriPaketi gonderilecekVeri;
float sayac = 0.0;

void setup() {
  Serial.begin(9600);
  
  // nRF24L01 başlatma kontrolü
  if (!radio.begin()) {
    Serial.println("HATA: nRF24L01 baslatilamadi! Baglantilari kontrol edin.");
    while (1);
  }

  // Alıcı ile aynı kanal (2422 MHz)
  radio.setChannel(22);

  // Yazma hattı (alıcının okuduğu adres)
  radio.openWritingPipe(address);
  
  // Sinyal gücü ayarı (RF24_PA_MIN, RF24_PA_LOW, RF24_PA_HIGH, RF24_PA_MAX)
  radio.setPALevel(RF24_PA_LOW);
  
  // Verici modunda dinleme kapatılır
  radio.stopListening();

  Serial.println("Verici hazir. Mesaj gonderimi basliyor...");
}

void loop() {
  // Gönderilecek verileri dolduruyoruz
  gonderilecekVeri.cihazID = 1;
  gonderilecekVeri.deger = sayac;
  strcpy(gonderilecekVeri.mesaj, " Merhaba Dunya");

  // Veriyi havadan gönder
  bool iletildi = radio.write(&gonderilecekVeri, sizeof(VeriPaketi));

  if (iletildi) {
    Serial.println("--- Paket Basariyla Gonderildi ---");
    Serial.print("Cihaz ID : ");
    Serial.println(gonderilecekVeri.cihazID);
    Serial.print("Deger    : ");
    Serial.println(gonderilecekVeri.deger);
    Serial.print("Mesaj    : ");
    Serial.println(gonderilecekVeri.mesaj);
    Serial.println("----------------------------------");
  } else {
    Serial.println("UYARI: Paket gonderilemedi! (Alici acik mi?)");
  }

  sayac += 1.0;
  
  // Her 1 saniyede bir gönder
  delay(1000);
}