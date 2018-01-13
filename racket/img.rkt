#lang racket

(require 2htdp/image)
(require 2htdp/universe)

;; simple align and overlay
(define vic
  (above (beside/align "bottom"
                       (triangle 40 "solid" "red")
                       (triangle 30 "solid" "red"))
         (rectangle 70 30 "solid" "black")))
(define door (rectangle 15 25 "solid" "brown"))
(define door-with-knob
  (overlay/align "right" "center" (circle 3 "solid" "yellow") door))
(overlay/align "center" "bottom" door-with-knob vic)


(define (a-number digit)
  (overlay
   (text (number->string digit) 12 "black")
   (circle 10 "solid" "white")))

;; rotation
(define (place-and-turn digit dial)
  (rotate 30
          (overlay/align "center" "top"
                         (above
                          (rectangle 1 5 "solid" "black")
                          (a-number digit))
                         dial)))

(place-and-turn 0 (circle 60 "solid" "black"))

(define (place-all-numbers dial)
  (foldl place-and-turn dial '(0 9 8 7 6 5 4 3 2 1)))


(define inner-dial
  (overlay
   (text "555-1234" 9 "black")
   (circle 30 "solid" "white")))

;; scale
(define (rotary-dial f)
  (scale
   f
   (overlay
    inner-dial
    (rotate
     -90
     (place-all-numbers (circle 60 "solid" "black"))))))

(rotary-dial 2)


;; blending, the myth is at the 4th argument of color. It actually
;; controls the opaque
(overlay
 (rectangle 60 100 "solid" (color 127 255 127 127))
 (rectangle 100 60 "solid" (color 127 127 255 127)))

;; spin, by recursive functions
(define (spin-alot t)
  (local [(define (spin-more i θ)
            (cond
              [(= θ 360) i]
              [else
               (spin-more (overlay i (rotate θ t))
                          (+ θ 1))]))]
    (spin-more t 0)))
(spin-alot (rectangle 12 120 "solid" (color 0 0 255 1)))
(spin-alot (triangle 120 "solid" (color 0 0 255 1)))
(spin-alot (isosceles-triangle 120 30 "solid" (color 0 0 255 1)))

(define (swoosh image s)
  (cond
    [(zero? s) image]
    [else (swoosh
           (overlay/align "center" "top"
                          (circle (* s 1/2) "solid" "yellow")
                          (rotate 4 image))
           (- s 1))]))
(swoosh (circle 100 "solid" "black") 100)


(define (carpet n)
  (cond
    [(zero? n) (square 1 "solid" "black")]
    [else
     (local [(define c (carpet (- n 1)))
             (define i (square (image-width c) "solid" "white"))]
       (above (beside c c c)
              (beside c i c)
              (beside c c c)))]))
(carpet 5)


(define (create-UFO-scene height)
  (underlay/xy (rectangle 100 100 "solid" "white") 50 height UFO))
 
(define UFO
  (underlay/align "center"
                  "center"
                  (circle 10 "solid" "green")
                  (rectangle 40 4 "solid" "green")))
 
(animate create-UFO-scene)
