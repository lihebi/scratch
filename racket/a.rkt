(module m racket
  (define x 10))

(require 'm)

#lang racket/base
(module box racket/base
  (provide (all-defined-out))
  (define b (box 0)))

(module transformers racket/base
  (provide (all-defined-out))
  (require (for-syntax racket/base
                       'box))
  (define-syntax (sett stx)
    (set-box! b 2)
    #'(void))
  (define-syntax (gett stx)
    #`#,(unbox b)))

(module user racket/base
  (provide (all-defined-out))
  (require 'transformers)
  (sett)
  (define gott (gett)))

(module test racket/base
  (require 'box 'transformers 'user)
  (displayln gott)
  (displayln (gett))
 
  (sett)
  (displayln (gett))
 
  (displayln (unbox b)))
(require 'test)


(let ((x 1))
  (let ((z 3))
    (define y 2)
    y)
  y
  )



(struct loc (line col))
(loc 1 3)
(loc-line (loc 1 3))

(define (foo (l))
  (loc? l))

(loc? '(1 2))
(loc? (loc 1 2))
