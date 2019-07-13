#lang racket
(require rackunit "solution1.rkt")

(check-equal? (r^match (r^range #\a #\d) "a") #\a)
(check-equal? (r^match (r^range #\a #\d) "b") #\b)
(check-false (r^match (r^range #\a #\d) "x"))
(check-false (r^match (r^range #\a #\d) "ax"))
(check-false (r^match (r^range #\a #\d) "xa"))

(check-equal? (r^match (r^or (r^range #\a #\d)
                             (r^range #\w #\z))
                       "b")
              #\b)
(check-equal? (r^match (r^or (r^range #\a #\d)
                             (r^range #\w #\z))
                       "x")
              #\x)
(check-false (r^match (r^or (r^range #\a #\d)
                            (r^range #\w #\z))
                      "m"))

(check-false (r^match (r^or) ""))

(check-equal? (r^match (r^seq (r^range #\a #\d)
                              (r^range #\p #\r)
                              (r^range #\w #\z))
                       "bqy")
              (vector-immutable #\b #\q #\y))
(check-equal? (r^match (r^seq (r^range #\a #\d)
                              (r^range #\p #\r)
                              (r^range #\w #\z))
                       "byq")
              #f)

(check-equal? (r^match (r^* (r^range #\a #\d) #\e) "ad") #\d)
(check-equal? (r^match (r^* (r^range #\a #\d) #\e) "") #\e)
(check-equal? (let ([l 0])
                (r^match (r^* (r^range #\a #\d) (begin (set! l (+ l 1)) #\q)) "")
                l)
              1)


(check-equal? (let ([l 0])
                (r^match (r^or (r^range #\x #\z)
                               (r^* (r^range #\a #\d) (begin (set! l (+ l 1)) #\q)))
                         "y")
                l)
              1)

;; (regexp-match "((([a-d])*)([x-z]))" "adz")
(check-equal? (r^match (r^seq (r^* (r^range #\a #\d) #\e)
                              (r^range #\x #\z))
                       "adz")
              (vector-immutable #\d #\z))
(check-equal? (r^match (r^seq (r^* (r^seq (r^range #\a #\b) (r^range #\c #\d))
                                   #\e)
                              (r^range #\x #\z))
                       "acbdz")
              (vector-immutable (vector-immutable #\b #\d) #\z))

(check-equal? (r^match (r^or (r^seq
                              (r^range #\a #\a)
                              (r^seq
                               (r^range #\b #\b)
                               (r^range #\c #\c)))
                             (r^seq
                              (r^seq
                               (r^range #\a #\a)
                               (r^range #\b #\b))
                              (r^range #\c #\c)))
                       "abc")
              (vector-immutable #\a (vector-immutable #\b #\c)))

(check-equal? (let ()
                (r^define x (r^range #\a #\d))
                (r^match (r^seq x x x) "abc"))
              (vector-immutable #\a #\b #\c))

(check-exn
 (λ (exn)
   (and (exn:fail:syntax? exn)
        (regexp-match
         (regexp-quote
          "identifiers bound with r^define must be used inside regexps")
         (exn-message exn))))
 (λ ()
   (expand
    #'(let ()
        (r^define x (r^range #\a #\c))
        x))))

(check-exn
 (λ (exn)
   (and (exn:fail:syntax? exn)
        (regexp-match
         (regexp-quote
          "expected identifier to be bound by r^define")
         (exn-message exn))))
 (λ ()
   (expand
    #'(let ()
        (define x 5)
        (r^match x "x")))))
