#lang racket

(require pict)
(require racket/draw)
(require file/convertible)

;; (filled-circle 1)

;; (filled-rectangle 30 30)
;; (rectangle  5 10)

(define pin-width 1)
(define pin-height 0.4)
(define pin-distance (- 0.65 pin-height))
(define inner-width 4.5)
(define inner-height 7.9)
(define outer-width 6.6)
(define circle-radius 0.1414)

(define (make-color-from-string s)
  "#f0f0f0"
  (let ([r (substring s 1 3)]
        [g (substring s 3 5)]
        [b (substring s 5 7)])
    (string->number s 16)
    (make-color (string->number r 16)
                (string->number g 16)
                (string->number b 16))))
;; (make-color-from-string "#f0f0f0")


(define pin
  (filled-rectangle pin-width pin-height
                    #:color (make-color-from-string "#f7bd13")
                    #:draw-border? #f))

(define col-pin (apply vl-append (cons pin-distance (make-list 12 pin))))

(define inner-rect
  (cc-superimpose
   (pin-over
    (rectangle inner-width inner-height
               #:border-width 0.127
               #:border-color (make-color-from-string "#f0f0f0"))
    0.5 0.5
    (circle circle-radius
            #:border-color (make-color-from-string "#f0f0f0")
            #:border-width 0.05))
   (rotate (text "TSSOP" null 2) (- (/ pi 2)))))


(module+ test
  (scale
   (hc-append col-pin inner-rect col-pin)
   ;; col-pin
   ;; inner-rect
   20))



(define (save-svg p filename)
  (let ([out (open-output-file filename
                               #:mode 'binary
                               #:exists 'replace)])
    (write-bytes (convert p 'svg-bytes)
                 out)
    (close-output-port out)))

(module+ test
  (save-svg (hc-append col-pin inner-rect col-pin)
            "out.svg")
  )

;; (define border (rectangle ))


;; (filled-rectangle 60 70 #:color "Thistle" #:border-color "Gainsboro" #:border-width 10)
