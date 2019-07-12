#lang racket/base

(require (for-syntax racket/base
                     syntax/parse
                     racket/list
                     racket/stxparam)
         racket/stxparam
         syntax/parse/define
         racket/contract
         racket/list
         syntax/parse
         syntax/macro-testing
         rackunit)

(+ 10
   (let/ec go-back-to-here
     (+ 1
        (+ 2
           (+ 3
              (go-back-to-here 7))))))

;; Exercise 16. Use let/ec to make a version of define that defines a
;; return statement.

(define-syntax (inner stx)
  (syntax-parse stx
    [(_ one other ...)
     #'(begin
         (inner one)
         (inner other ...))]))

(define-syntax (define-w/return stx)
  (syntax-parse stx
    [(_ out (fun x ...)
        before ...
        ((~literal return) v)
        after ...)
     #'(define (fun x ...)
         (let/ec return
           
           ;; (let-syntax
           ;;     ([inner (λ (stx)
           ;;               (syntax-parse stx
           ;;                 [(_ one other ...)
           ;;                  #'(begin
           ;;                      (inner one)
           ;;                      (inner other ...))]
           ;;                 [(_ ((~literal return) v) other ...)
           ;;                  #'(out v)]))]))
           
           before ...
           (out v)))]
    [(_ (fun x ...) body ...)
     #'(define (fun x ...)
         body ...)]))


(define-syntax-parameter return
  (λ (stx) (raise-syntax-error
            'return "Illegal use of return outside function" stx)))

(define-syntax (define-w/return-2 stx)
  (syntax-parse stx
    [(_ (fun x ...) body ...+)
     #;
     (syntax-parameterize
         ([return (make-rename-transformer #'out)])
       #'(define (fun x ...)
           (let/ec out
             body ...
             )))
     
     #'(define (fun x ...)
         (let/ec out
           (syntax-parameterize
               ([return (make-rename-transformer #'out)])
             body ...)))]))

(define-w/return-2 (foo)
  (+ 1
     (+ 2
        (return 3))))
(foo)


(define-syntax (define-w/return-3 stx)
  (syntax-parse stx
    [(_ (fun x ...) body ...+)
     #'(define (fun x ...)
         (let/ec return
           body ...))]))

(define (bar)
  (let/ec return
    (+ 1
       (+ 2
          (return 3)))))

#;
(define-w/return-3 (foo)
  (+ 1
     (+ 2
        (return 3))))
;; (bar)
;; (foo)

#;
(define-w/return-2 (foo x)
  (+ 1 2)
  ;; (return 5)
  (when (< 2 3)
    (return 5))
  (/ 1 0))

;; (foo 3)

;; usage
#;
(define-w/return out (foo x)
  (+ 1 2)
  ;; (return 5)
  (when (< 2 3)
    (return 5))
  (/ 1 0))

;; (foo 3)

;; Exercise 17. Use let/ec to make a while macro that supports break
;; and continue.

(define-syntax (my-while-wrong stx)
  (syntax-parse stx
    [(_ c before ... (~literal break) after ...)
     #'(let/ec out
         (while c before ...))]
    [(_ c before ...)
     #'(let/ec out
         (while c before ...))]))

(while (< i 3)
  (displayln i)
  (add1 i))

(define-syntax (my-while stx)
  (syntax-parse stx
    [(_ c body ...)
     #'(let/ec out
         (syntax-parameterize
             ([return (make-rename-transformer #'out)])
           body ...))]))

;; usage

(define ct 0)
(my-while (< ct 5)
          (set! ct (add1 ct))
          (when (< ct 3)
            break))


(define-syntax (my-while-1 stx)
  (syntax-parse stx
    [(_ c body ...)
     #'(let/ec break
         (begin
           (let/ec continue
             body ...
             (if c
                 (continue)
                 (break)))))]))

(define ct 0)
(my-while-1 (< ct 7)
          (set! ct (add1 ct))
          (when (< ct 2)
            continue)
          (displayln "hello")
          (when (< ct 5)
            break))



(let/ec break
  (let/ec continue
    (when ct)))


