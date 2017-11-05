#lang racket

;; download youtube music


;; cmd: youtube-dl --extract-audio --audio-format flac <url>

(define (youtube-dl-single url f)
  (let ([cmd (string-append "youtube-dl"
                            " --extract-audio"
                            " --audio-format flac"
                            " --output \"" f ".flac\""
                            " "
                            url)])
    (displayln cmd)
    (printf "Downloading ~a~n" f)
    (system cmd)
    (displayln "Downloaded")))

(define (youtube-dl pairs)
  (for ([pair pairs])
    (youtube-dl-single (car pair) (cdr pair))))

(module+ test
  (youtube-dl
   '(("https://www.youtube.com/watch?v=s-CcFyyPJiY&index=2&list=RDvTtcFMY7-vI" . "張杰 - 三生三世")
     ("https://www.youtube.com/watch?v=vTtcFMY7-vI&list=RDvTtcFMY7-vI&t=2" . "杨宗纬&张碧晨 - 涼涼")
     ("https://www.youtube.com/watch?v=c68lZi7Kej8&list=RDvTtcFMY7-vI&index=3" . "郁可唯 - 思慕")
     ("https://www.youtube.com/watch?v=QMi9DQPTIa8&list=RDvTtcFMY7-vI&index=4" . "董貞 - 繁花")
     ("https://www.youtube.com/watch?v=9NIrgZwwOcY&list=PLfkWGvionty8hx8e6yz9gDbehWAccLVKb&index=3" . "香香 - 就算沒有如果"))))
