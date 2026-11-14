#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>

// Pin Tanımlamaları
#define CE_PIN 7
#define CSN_PIN 8

RF24 radio(CE_PIN, CSN_PIN);
const byte address[6] = "00001";

// Verici ile BİREBİR AYNISI olması gereken veri paketi
struct VeriPaketi {
  byte cihazID;        // Gönderen cihazın numarası
  float deger;         // Gelen sayısal değer / sayaç
  char mesaj[20];      // Gelen metin
};

VeriPaketi gelenVeri;

void setup() {
  Serial.begin(9600);

  // nRF24L01 başlatma kontrolü
  if (!radio.begin()) {
    Serial.println("HATA: nRF24L01 baslatilamadi! Baglantilari kontrol edin.");
    while (1);
  }

  // Verici ile aynı kanal (2422 MHz)
  radio.setChannel(22);

  // Okuma hattı (vericinin yazdığı adres, 0 numaralı pipe üzerinden dinlenir)
  radio.openReadingPipe(0, address);

  // Sinyal gücü ayarı
  radio.setPALevel(RF24_PA_LOW);

  // Alıcı modunda dinleme başlatılır
  radio.startListening();

  Serial.println("Alici hazir. Veri bekleniyor...");
}

void loop() {
  // Havadan gelen yeni bir veri paketi var mı kontrol et
  if (radio.available()) {
    // Gelen veriyi struct yapısının içine oku
    radio.read(&gelenVeri, sizeof(VeriPaketi));

    Serial.println("--- Yeni Paket Alindi ---");
    Serial.print("Cihaz ID : ");
    Serial.println(gelenVeri.cihazID);
    Serial.print("Deger    : ");
    Serial.println(gelenVeri.deger);
    Serial.print("Mesaj    : ");
    Serial.println(gelenVeri.mesaj);
    Serial.println("-------------------------");
  }
}