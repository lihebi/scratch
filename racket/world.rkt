#lang racket

(require 2htdp/planetcute)
(require 2htdp/image)

(define (stack imgs)
  (cond
    [(empty? (rest imgs)) (first imgs)]
    [else (overlay/xy (first imgs)
                      0 40
                      (stack (rest imgs)))]))


(beside/align
   "bottom"
   (stack (list wall-block-tall stone-block))
   (stack (list character-cat-girl
                stone-block stone-block
                stone-block stone-block))
   water-block
   (stack (list grass-block dirt-block))
   (stack (list grass-block dirt-block dirt-block)))

