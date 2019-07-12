#lang s-exp "pfsh5.rkt"

;; (run ls -l)
(run whoami)
;; (define me (run whoami))
;; me
;; me
;; (string-append me "hello")


;; (define l (run ls))
;; l
;; me
;; (run wc -l < l)

;; (run ls "-1")

(&& (run test -f demok.txt)
    (run cat demo.txt))

(&& (&& (run cat demo.txt)
        (run cat demo.txt))
    (run test -f demo.txt))


