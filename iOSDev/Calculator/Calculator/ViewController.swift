//
//  ViewController.swift
//  Calculator
//
//  Created by Hoang Nguyen Thai on 8/28/16.
//  Copyright © 2016 Hoang NT. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet private weak var display: UILabel!
    
    private var userIsInTheMiddleOfTyping = false
    private var activeFloatingPoint = false
    
    @IBAction private func touchDigit(sender: UIButton) {
        let digit = sender.currentTitle!
        if userIsInTheMiddleOfTyping {
            let textCurrentlyInDisplay = display.text!
            display.text = textCurrentlyInDisplay + digit
        } else {
            display.text = digit
        }
        userIsInTheMiddleOfTyping = true
    }
    @IBAction private func floatingPoint(sender: UIButton) {
        if !activeFloatingPoint {
            if userIsInTheMiddleOfTyping {
                activeFloatingPoint = true
                let textCurrentlyInDisplay = display.text!
                display.text = textCurrentlyInDisplay + sender.currentTitle!
            } else {
                display.text = "0" + sender.currentTitle!
            }
        }
        userIsInTheMiddleOfTyping = true
    }
    
    // computed property
    private var displayValue: Double {
        get {
            return Double(display.text!)!
        }
        set {
            if (Double(Int(newValue)) == newValue) {
                display.text = String(Int(newValue))
            } else {
                display.text = String(newValue)
            }
        }
    }
    
    private var brain = CalculatorBrain()

    @IBAction private func performOperation(sender: UIButton) {
        activeFloatingPoint = false
        if userIsInTheMiddleOfTyping {
            brain.setOperand(displayValue)
            userIsInTheMiddleOfTyping = false
        }
        if let mathematicalSymbol = sender.currentTitle {
            brain.performOperation(mathematicalSymbol)
        }
        displayValue = brain.result
    }
}