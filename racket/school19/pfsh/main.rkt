#lang racket
(require "run.rkt"
         racket/port
         (for-syntax syntax/parse
                     racket/list)
         syntax/wrap-modbeg)
 
(provide
 ;; #%module-begin
 > <
 (rename-out [pfsh:run #%app]
             [pfsh:top #%top]
             [pfsh:define define]
             [pfsh:datum #%datum]
             [my-module-begin #%module-begin]
             [pfsh:top-interaction #%top-interaction]
             [pfsh:string-append string-append]))

#;
(define-syntax (my-wrapper stx)
  (syntax-parse stx
    [(_ e)
     ;; 1. check whether on the same line
     ;; 2. check whether there's explicit parenthesis
     ;; 2.1 add parenthesis if necessary
     (if (pair? (syntax-e #'e)) #'e
         ;; read everything up to the end of this line
         ;;
         ;; If datum consumed, would it be removed from the input
         ;;stream?
         (let ([l (syntax-line #'e)])
           (datum->syntax
            (for/list ([i
                        (in-naturals)
                        ;; (in-range 1)
                        ])
              ;; (define d (read))
              (define d #'e)
              (displayln d)
              #:break (not (equal? l (syntax-line d)))
              d))))]))

#;
(define-syntax my-module-begin
  (make-wrapping-module-begin
   #'my-wrapper
   ))

;; (apply append '((1 2) (3)))

(define-syntax (my-module-begin stx)
  (let ([l (syntax->list stx)])
    (let ([sorted (group-by syntax-line l)])
      (datum->syntax #f
       (apply append (for/list ([li sorted])
                       (if (pair? (syntax-e (first li))) li
                           (list li))))))))
 
(module reader syntax/module-reader
  pfsh
  #:wrapper1
  (λ (thunk)
    ;; the thunk returns a syntax object for every call?
    (thunk)))
 
(define-syntax (pfsh:run stx)
  (syntax-parse stx
    ;; (HEBI: ??)
    ;; #:datum-literals (< >)
    #:literals (< >)
    [(_ prog arg ... < stream:expr)
     #'(with-input-from-string
         stream
         (lambda ()
           (pfsh:run prog arg ...)))]
    [(_ prog arg ... > out:id)
     #'(define out (with-output-to-string
                     (lambda ()
                       (pfsh:run prog arg ...))))]
    [(_ prog arg ... < stream:expr > out:id)
     #'(define out (with-output-to-string
                     (with-input-from-string
                       stream
                       (lambda ()
                         (pfsh:run prog arg ...)))))]
    [(_ prog arg ...)
     #`(void (run prog arg ...))]))

#;
(define-syntax >
  (raise-syntax-error #f "Bad syntax of using >" #'here))

(define-syntax (> stx)
  (raise-syntax-error #f "Bad syntax of using >" stx))
(define-syntax (< stx)
  (raise-syntax-error #f "Bad syntax of using <" stx))
 
(define-syntax (pfsh:top stx)
  (syntax-parse stx
    [(_ . sym:id)
     #`#,(symbol->string (syntax-e #'sym))]))
 
(define-syntax (pfsh:define stx)
  (syntax-parse stx
    [(_ stream:id expr)
     #'(define stream expr)]
    [(_ (proc:id arg:id ...) expr)
     #'(begin
         (define (actual-proc arg ...)
           expr)
         (define-syntax (proc stx)
           (syntax-parse stx
             [(_ arg ...) #'(actual-proc arg ...)])))]))
 
(pfsh:define (pfsh:string-append arg1 arg2)
             (string-append arg1 arg2))
 
(define-syntax (pfsh:datum stx)
  (syntax-parse stx
    [(_ . s:string) #'(#%datum . s)]
    [(_ . other)
     (raise-syntax-error 'pfsh
                         "only literal strings are allowed"
                         #'other)]))
 
(define-syntax (pfsh:top-interaction stx)
  (syntax-parse stx
    [(_ . form) #'form]))
