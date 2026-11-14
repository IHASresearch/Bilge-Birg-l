/*
 * ESP8266 Wi-Fi Paket Düşürme & Dayanıklılık Testi - SERVER (Ana Cihaz)
 * Görev: SoftAP ağı kurar, 4210 UDP portunu dinler ve SEQ numaralarından kayıp paketleri hesaplar.
 */

#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

// Server Access Point Ayarları (Client ile BİREBİR AYNI)
const char* AP_SSID = "ESP8266_DROP_TEST";
const char* AP_PASS = "12345678";

const unsigned int LOCAL_PORT = 4210;

WiFiUDP Udp;
char incomingPacket[255];

unsigned long lastReceivedSeq = 0;      // En son alınan paketin sıra numarası
unsigned long totalReceived = 0;        // Başarıyla alınan toplam paket sayısı
unsigned long totalLostDetected = 0;    // Arada kaybolduğu tespit edilen paket sayısı

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH); // LED Kapalı (ESP8266 ters mantık)

  Serial.println("\n==================================================");
  Serial.println("   ESP8266 WIFI PAKET TEST SERVER (ANA CİHAZ)");
  Serial.println("==================================================");

  // Erişim Noktası (SoftAP) modunu başlat
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);

  IPAddress myIP = WiFi.softAPIP();
  Serial.println("[✓] Wi-Fi Erişim Noktası (AP) Başlatıldı!");
  Serial.print("    Ağ Adı (SSID) : "); Serial.println(AP_SSID);
  Serial.print("    Server IP     : "); Serial.println(myIP); // Varsayılan: 192.168.4.1

  // UDP Dinlemeyi Başlat
  Udp.begin(LOCAL_PORT);
  Serial.print("    Dinlenen Port : "); Serial.println(LOCAL_PORT);
  Serial.println("--------------------------------------------------");
  Serial.println("Client bağlantısı ve UDP paketleri bekleniyor...\n");
}

void loop() {
  int packetSize = Udp.parsePacket();

  if (packetSize > 0) {
    int len = Udp.read(incomingPacket, sizeof(incomingPacket) - 1);
    if (len > 0) {
      incomingPacket[len] = '\0'; // String sonlandırma karakteri
    }

    totalReceived++;

    // Gelen mesaj formatı: "SEQ:1|TIME:1234|DATA:PING_TEST"
    // Başındaki "SEQ:" kısmından sıra numarasını ayıkla
    unsigned long currentSeq = 0;
    if (strncmp(incomingPacket, "SEQ:", 4) == 0) {
      currentSeq = strtoul(incomingPacket + 4, NULL, 10);
    }

    // Paket kaybı (atlama) kontrolü
    unsigned long missedInThisStep = 0;
    if (lastReceivedSeq > 0 && currentSeq > lastReceivedSeq + 1) {
      missedInThisStep = currentSeq - lastReceivedSeq - 1;
      totalLostDetected += missedInThisStep;

      Serial.printf("[KAYIP TESPİTİ] #%lu ile #%lu arasında %lu paket kayboldu!\n",
                    lastReceivedSeq, currentSeq, missedInThisStep);
    } else if (currentSeq == 1 && lastReceivedSeq > 1) {
      // Client resetlendiyse sayaçları sıfırla
      Serial.println("\n[BİLGİ] Client yeniden başlatıldı, sayaçlar sıfırlanıyor...\n");
      totalReceived = 1;
      totalLostDetected = 0;
    }

    lastReceivedSeq = currentSeq;

    Serial.printf("[RECV] Paket #%lu Alındı (%s) | Gönderen: %s:%d\n",
                  currentSeq, incomingPacket, Udp.remoteIP().toString().c_str(), Udp.remotePort());

    // Paket alındığında LED'i kısa süre yak
    digitalWrite(LED_BUILTIN, LOW);
    delay(30);
    digitalWrite(LED_BUILTIN, HIGH);

    // Her 10 alınan pakette bir özet kayıp istatistiği yazdır
    if (totalReceived % 10 == 0) {
      unsigned long expectedTotal = totalReceived + totalLostDetected;
      float lossRate = (expectedTotal > 0) ? ((float)totalLostDetected / expectedTotal) * 100.0 : 0.0;

      Serial.println("\n--- [SERVER ALIM & KAYIP RAPORU] ---");
      Serial.printf("  Son Alınan SEQ     : #%lu\n", currentSeq);
      Serial.printf("  Toplam Alınan      : %lu\n", totalReceived);
      Serial.printf("  Tespit Edilen Kayıp: %lu\n", totalLostDetected);
      Serial.printf("  Hesaplanan Kayıp   : %.1f%%\n", lossRate);
      Serial.println("------------------------------------\n");
    }
  }

  yield();
}