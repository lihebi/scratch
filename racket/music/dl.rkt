#lang racket

;; download youtube music


;; cmd: youtube-dl --extract-audio --audio-format flac <url>

(define (youtube-dl-single url f (form "flac"))
  (let ([cmd (string-append "youtube-dl"
                            " --extract-audio"
                            " --audio-format " form
                            " --output \"" f "." form "\""
                            " "
                            url)])
    (displayln cmd)
    (printf "Downloading ~a~n" f)
    (system cmd)
    (displayln "Downloaded")))

(define (youtube-dl pairs)
  (for ([pair pairs])
    (youtube-dl-single (car pair) (cdr pair) "mp3")))

(module+ test
  (youtube-dl
   '(("https://www.youtube.com/watch?v=s-CcFyyPJiY&index=2&list=RDvTtcFMY7-vI" . "張杰 - 三生三世")
     ("https://www.youtube.com/watch?v=vTtcFMY7-vI&list=RDvTtcFMY7-vI&t=2" . "杨宗纬&张碧晨 - 涼涼")
     ("https://www.youtube.com/watch?v=c68lZi7Kej8&list=RDvTtcFMY7-vI&index=3" . "郁可唯 - 思慕")
     ("https://www.youtube.com/watch?v=QMi9DQPTIa8&list=RDvTtcFMY7-vI&index=4" . "董貞 - 繁花")
     ("https://www.youtube.com/watch?v=9NIrgZwwOcY&list=PLfkWGvionty8hx8e6yz9gDbehWAccLVKb&index=3" . "香香 - 就算沒有如果")))

  (youtube-dl
   '(("https://www.youtube.com/watch?v=JGwWNGJdvx8" . "Ed Sheeran - Shape of You")
     ("https://www.youtube.com/watch?v=lp-EO5I60KA" . "Ed Sheeran - Thinking Out Loud")
     ("https://www.youtube.com/watch?v=V54CEElTF_U" . "Taylor Swift - Call It What You Want")
     ("https://www.youtube.com/watch?v=8xg3vE8Ie_E" . "Taylor Swift - Love Story")
     ("https://www.youtube.com/watch?v=VuNIsY6JdUw" . "Taylor Swift - You Belong With Me")
     ("https://www.youtube.com/watch?v=AJtDXIazrMo&index=10&list=RD8xg3vE8Ie_E" . "Ellie Goulding - Love Me Like You Do")
     ("https://www.youtube.com/watch?v=Zlot0i3Zykw&index=11&list=RD8xg3vE8Ie_E" . "Taylor Swift - Red")
     ("https://www.youtube.com/watch?v=AgFeZr5ptV8&list=RD8xg3vE8Ie_E&index=15" . "Taylor Swift - 22")
     ("https://www.youtube.com/watch?v=xKCek6_dB0M&list=RD8xg3vE8Ie_E&index=16" . "Taylor Swift - Teardrops On My Guitar")
     ("https://www.youtube.com/watch?v=nfWlot6h_JM&list=RD8xg3vE8Ie_E&index=23" . "Taylor Swift - Shake It Off")
     ("https://www.youtube.com/watch?v=Bg59q4puhmg&list=RD8xg3vE8Ie_E&index=47" . "Avril Lavigne - Girlfriend")
     ("https://www.youtube.com/watch?v=5NPBIwQyPWE" . "Avril Lavigne - Complicated")
     ("https://www.youtube.com/watch?v=8xoG0Xv3vs0" . "Avril Lavigne - Innocence")))

  (youtube-dl-single "https://www.youtube.com/watch?v=5NPBIwQyPWE" "aaa" "mp3")

  )
